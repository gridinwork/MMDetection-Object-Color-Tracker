"""Checkpoint downloads with progress callbacks."""

from __future__ import annotations

import urllib.request
from pathlib import Path


class DownloadError(RuntimeError):
    pass


def download_file(url: str, dest: Path, progress=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MMDetectionVisionStudio/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise DownloadError(f"Download failed: {exc}") from exc
    temporary.replace(dest)


def checkpoint_looks_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 500_000:
        return False
    with path.open("rb") as handle:
        magic = handle.read(2)
    return magic in (b"PK", b"\x80\x02", b"\x80\x03", b"\x80\x04")
