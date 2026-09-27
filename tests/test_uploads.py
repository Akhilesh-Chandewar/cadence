"""Upload feature tests: store rules, resolution safety, API round trips."""

from __future__ import annotations

import pytest

from cadence.api.uploads import (
    UploadError,
    resolve_source_path,
    resolve_upload,
    save_upload,
)


@pytest.fixture(autouse=True)
def _upload_dir(tmp_path, monkeypatch):
    """Redirect the uploads store to a temp dir (module attr + env for children)."""
    monkeypatch.setenv("CADENCE_UPLOAD_DIR", str(tmp_path / "uploads"))
    from cadence.api import uploads

    monkeypatch.setattr(uploads, "UPLOAD_DIR", tmp_path / "uploads")
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from cadence.api.main import app

    return TestClient(app)


class TestSaveUpload:
    def test_saves_csv_and_returns_id(self):
        upload_id = save_upload("data.csv", b"unique_id,ds,y\na,2024-01-01,1\n")
        assert len(upload_id) == 32
        assert resolve_upload(upload_id).suffix == ".csv"

    def test_rejects_bad_type(self):
        with pytest.raises(UploadError, match="unsupported file type"):
            save_upload("evil.exe", b"MZ...")

    def test_rejects_empty(self):
        with pytest.raises(UploadError, match="empty"):
            save_upload("data.csv", b"")

    def test_size_cap(self, monkeypatch):
        monkeypatch.setenv("CADENCE_MAX_UPLOAD_MB", "1")
        with pytest.raises(UploadError, match="too large"):
            save_upload("big.csv", b"x" * (1024 * 1024 + 1))

    def test_parquet_suffix_allowed(self):
        upload_id = save_upload("data.PARQUET", b"parquet-bytes")  # case-insensitive
        assert resolve_upload(upload_id).suffix == ".parquet"


class TestResolve:
    def test_unknown_id(self):
        with pytest.raises(UploadError, match="unknown upload_id"):
            resolve_upload("0" * 32)

    def test_traversal_and_injection_rejected(self):
        for bad in ["../../etc/passwd", "a/b", "x;y", "%2e%2e", ""]:
            with pytest.raises(UploadError, match="malformed"):
                resolve_upload(bad)

    def test_resolve_source_path_fills_path(self):
        upload_id = save_upload("d.csv", b"unique_id,ds,y\na,2024-01-01,1\n")
        source = resolve_source_path({"upload_id": upload_id, "horizon": 1})
        assert source["path"].endswith(".csv")

    def test_resolve_rejects_both_id_and_path(self):
        with pytest.raises(UploadError, match="not both"):
            resolve_source_path({"upload_id": "abc", "path": "/x.csv"})


class TestAPIRoundTrip:
    def test_upload_then_forecast(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        with open(tmp_path / "s.csv", "rb") as fh:
            resp = client.post("/data/upload", files={"file": ("seasonal.csv", fh, "text/csv")})
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]

        forecast = client.post(
            "/forecast",
            json={"source_config": {"upload_id": upload_id}, "horizon": 6},
        )
        assert forecast.status_code == 200
        assert "synthetic_seasonal" in forecast.json()["report"]

    def test_upload_rejects_exe(self, client):
        resp = client.post(
            "/data/upload",
            files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "unsupported file type" in resp.json()["detail"]

    def test_stream_accepts_upload_id(self, client, tmp_path):
        import shutil

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        with open(tmp_path / "s.csv", "rb") as fh:
            upload_id = client.post(
                "/data/upload", files={"file": ("s.csv", fh, "text/csv")}
            ).json()["upload_id"]

        with client.stream(
            "GET", "/pipeline/stream", params={"upload_id": upload_id, "horizon": 6}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert '"stage"' in body  # five-node checkpoints streamed
