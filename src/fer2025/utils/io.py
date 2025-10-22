from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/fer2025"))


@dataclass(frozen=True)
class RemoteFile:
    url: str
    checksum: Optional[str]
    checksum_type: str = "sha256"
    filename: Optional[str] = None


def ensure_cache_dir(path: Optional[str] = None) -> Path:
    cache_dir = Path(os.path.expanduser(path)) if path else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download(remote: RemoteFile, destination: Path) -> None:
    with requests.get(remote.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}")
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        progress.close()


def _verify_checksum(path: Path, checksum: str, checksum_type: str = "sha256") -> bool:
    if not checksum:
        return True
    try:
        hasher = hashlib.new(checksum_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported checksum type: {checksum_type}") from exc

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return digest.lower() == checksum.lower()


def ensure_file(remote: RemoteFile, cache_dir: Optional[str] = None) -> Path:
    directory = ensure_cache_dir(cache_dir)
    target_name = remote.filename or Path(remote.url).name
    target = directory / target_name

    if target.exists():
        if remote.checksum and not _verify_checksum(target, remote.checksum, remote.checksum_type):
            target.unlink()
        else:
            return target

    _download(remote, target)

    if remote.checksum and not _verify_checksum(target, remote.checksum, remote.checksum_type):
        target.unlink(missing_ok=True)
        raise ValueError(f"Checksum verification failed for {target}")

    return target


def dump_metrics(metrics_path: Path, frames: list[dict]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2)
