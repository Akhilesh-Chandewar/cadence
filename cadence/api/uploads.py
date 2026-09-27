"""Upload handling for production use: validated storage + source_config resolution.

Uploads land in a single directory (env-configurable) with an opaque
``upload_id`` filename; API and UI resolve ``upload_id`` back to a concrete
path before handing a source_config to the graph. Rules: CSV/Parquet only,
hard size cap (env ``CADENCE_MAX_UPLOAD_MB``, default 200), unique ids —
no user-controlled paths ever reach the filesystem.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

UPLOAD_DIR = Path(os.environ.get("CADENCE_UPLOAD_DIR", "/tmp/cadence-uploads"))
DEFAULT_MAX_MB = 200
ALLOWED_SUFFIXES = {".csv", ".parquet"}


class UploadError(ValueError):
    """Raised for invalid uploads (type, size, unknown id)."""


def max_upload_bytes() -> int:
    return int(os.environ.get("CADENCE_MAX_UPLOAD_MB", DEFAULT_MAX_MB)) * 1024 * 1024


def save_upload(filename: str, data: bytes) -> str:
    """Validate and persist an upload; returns its opaque upload_id."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(
            f"unsupported file type {suffix or '(none)'!r}; upload a .csv or .parquet file"
        )
    if not data:
        raise UploadError("uploaded file is empty")
    cap = max_upload_bytes()
    if len(data) > cap:
        raise UploadError(
            f"file too large ({len(data) / 1024 / 1024:.1f} MB); cap is {cap // 1024 // 1024} MB "
            "(env CADENCE_MAX_UPLOAD_MB)"
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    (UPLOAD_DIR / f"{upload_id}{suffix}").write_bytes(data)
    return upload_id


def resolve_upload(upload_id: str) -> Path:
    """Map an upload_id to its stored file path, safely (no traversal possible:
    ids are validated hex, and only our directory is searched)."""
    clean = (upload_id or "").strip()
    if not clean or len(clean) > 64 or any(c not in "0123456789abcdef" for c in clean):
        raise UploadError(f"malformed upload_id {upload_id!r}")
    for suffix in sorted(ALLOWED_SUFFIXES):
        candidate = UPLOAD_DIR / f"{clean}{suffix}"
        if candidate.exists():
            return candidate
    raise UploadError(f"unknown upload_id {upload_id!r} (it may have expired)")


def resolve_source_path(source_config: dict) -> dict:
    """Fill in 'path' from 'upload_id' when present (mutates a copy)."""
    source = dict(source_config or {})
    upload_id = source.pop("upload_id", None)
    if upload_id:
        if "path" in source:
            raise UploadError("pass either upload_id or path, not both")
        source["path"] = str(resolve_upload(upload_id))
    return source
