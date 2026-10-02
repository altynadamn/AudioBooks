from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from audiobooks.api.app import create_app
from audiobooks.services.context import AppContext
from audiobooks.services.pipeline import Pipeline


@pytest.fixture
def client(ctx: AppContext) -> TestClient:
    app = create_app(ctx)
    # run jobs inline so the test can observe the result deterministically
    app.state.runner.submit = lambda job_id: Pipeline(ctx).run(job_id)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["voices_available"] == 5


def test_upload_analyze_and_characters(client: TestClient, sample_txt: Path) -> None:
    with sample_txt.open("rb") as fh:
        r = client.post("/books", files={"file": ("../../evil name.txt", fh, "text/plain")})
    assert r.status_code == 201, r.text
    book_id = r.json()["id"]

    r = client.post(f"/books/{book_id}/analyze", json={})
    assert r.status_code == 202
    job = client.get(f"/jobs/{r.json()['id']}").json()
    assert job["status"] == "done"

    chars = client.get(f"/books/{book_id}/characters").json()
    assert {c["id"] for c in chars} == {"anna", "ivan"}

    r = client.put(f"/books/{book_id}/characters/anna/voice", json={"voice_id": "female_02"})
    assert r.status_code == 200 and r.json()["voice_locked"] is True

    script = client.get(f"/books/{book_id}/chapters/1/script").json()
    assert script["analysis"] == "llm"
    assert len(client.get(f"/books/{book_id}/chapters").json()) == 2


def test_rejects_unsupported_upload(client: TestClient) -> None:
    r = client.post("/books", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_unknown_ids_return_404(client: TestClient) -> None:
    assert client.get("/books/nope").status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    assert client.put("/books/nope/characters/x/voice", json={"voice_id": "v"}).status_code == 404
