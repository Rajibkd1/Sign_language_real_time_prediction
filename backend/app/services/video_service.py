from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from app.config import Settings
from app.errors import InvalidUploadError, UploadTooLargeError


ALLOWED_VIDEO_EXTENSIONS = {".webm", ".mp4", ".mov", ".m4v", ".avi"}
ALLOWED_VIDEO_MIME_PREFIXES = ("video/", "application/octet-stream")


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    original_filename: str
    content_type: str
    size_bytes: int


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return suffix
    return ".webm"


def validate_upload_metadata(upload: UploadFile) -> None:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (upload.content_type or "").lower()

    if suffix and suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise InvalidUploadError(
            "Unsupported video extension.",
            detail=f"Allowed extensions: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
        )

    if content_type and not any(content_type.startswith(prefix) for prefix in ALLOWED_VIDEO_MIME_PREFIXES):
        raise InvalidUploadError("Unsupported upload MIME type.", detail=content_type)


async def store_upload_to_temp(upload: UploadFile, settings: Settings) -> StoredUpload:
    validate_upload_metadata(upload)
    suffix = _safe_suffix(upload.filename or "recording.webm")
    tmp = tempfile.NamedTemporaryFile(prefix="bdsl_", suffix=suffix, delete=False)
    path = Path(tmp.name)
    size = 0
    try:
        with tmp:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadTooLargeError(
                        detail=f"Upload exceeded {settings.max_upload_mb} MB."
                    )
                tmp.write(chunk)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
    finally:
        await upload.close()

    if size <= 0:
        path.unlink(missing_ok=True)
        raise InvalidUploadError("The upload was empty.")

    return StoredUpload(
        path=path,
        original_filename=upload.filename or "recording",
        content_type=upload.content_type or "application/octet-stream",
        size_bytes=size,
    )


def delete_temp_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

