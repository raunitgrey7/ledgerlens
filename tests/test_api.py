import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from ledgerlens.api.app import create_app
from ledgerlens.api.storage import Storage
from tests.conftest import make_settings, png_bytes


def test_health_and_metrics(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok" and health.json()["ocr_engine"] == "atlas"
    assert health.headers["X-Request-ID"]
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and "ledgerlens_http_requests_total" in metrics.text
    assert "ledgerlens_review_queue_pending" in metrics.text


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_ingest_get_and_queue_flow(client: TestClient, page_by_type) -> None:
    from ledgerlens.schemas import DocType

    page, _truth = page_by_type[DocType.INVOICE]
    response = client.post("/v1/documents", files={"file": ("inv.png", png_bytes(page), "image/png")})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["result"]["doc_type"] == "invoice"
    assert any(f["name"] == "total" for f in body["result"]["fields"])
    fetched = client.get(f"/v1/documents/{body['id']}")
    assert fetched.status_code == 200 and fetched.json()["id"] == body["id"]
    queue = client.get("/v1/queue").json()
    assert queue["pending"] == (1 if body["status"] == "pending" else 0)
    stats = client.get("/v1/stats").json()
    assert stats["documents"] == 1


def test_review_records_corrections_and_effort(client: TestClient, page_by_type) -> None:
    from ledgerlens.schemas import DocType

    page, _ = page_by_type[DocType.RECEIPT]
    doc = client.post("/v1/documents", files={"file": ("r.png", png_bytes(page), "image/png")}).json()
    predicted = {f["name"]: f["value"] for f in doc["result"]["fields"]}
    corrections = dict(predicted)
    corrections["total"] = "999999.00"
    review = client.post(f"/v1/queue/{doc['id']}/review", json={"reviewer": "ana", "fields": corrections})
    assert review.status_code == 200, review.text
    body = review.json()
    assert body["fields_corrected"] == 1 and body["fields_confirmed"] == len(predicted) - 1
    assert 0 < body["effort_saved_fraction"] < 1
    stats = client.get("/v1/stats").json()
    assert stats["reviewed"] == 1 and stats["fields_corrected"] == 1
    assert client.get(f"/v1/documents/{doc['id']}").json()["status"] == "reviewed"


def test_review_validation_and_not_found(client: TestClient) -> None:
    assert client.get("/v1/documents/nope").status_code == 404
    assert client.post("/v1/queue/nope/review", json={"fields": {}}).status_code == 404


def test_upload_validation(client: TestClient) -> None:
    assert client.post("/v1/documents", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 415
    assert client.post("/v1/documents", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    tiny = Image.new("RGB", (10, 10), "white")
    assert client.post("/v1/documents", files={"file": ("t.png", png_bytes(tiny), "image/png")}).status_code == 422
    gif = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(gif, format="GIF")
    assert client.post("/v1/documents", files={"file": ("g.gif", gif.getvalue(), "image/gif")}).status_code == 415


def test_upload_size_limit(engine, tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path, max_upload_bytes=500), engine=engine, storage=Storage(":memory:"))
    client = TestClient(app)
    big = png_bytes(Image.effect_noise((300, 300), 80).convert("RGB"))
    assert len(big) > 500
    assert client.post("/v1/documents", files={"file": ("big.png", big, "image/png")}).status_code == 413


def test_api_key_protects_v1_routes(engine, tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path, api_key="secret"), engine=engine, storage=Storage(":memory:"))
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/stats").status_code == 401
    assert client.get("/v1/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/stats", headers={"X-API-Key": "secret"}).status_code == 200


def test_review_rejects_unknown_fields(client: TestClient, page_by_type) -> None:
    from ledgerlens.schemas import DocType

    page, _ = page_by_type[DocType.INVOICE]
    doc = client.post("/v1/documents", files={"file": ("i.png", png_bytes(page), "image/png")}).json()
    bad = client.post(f"/v1/queue/{doc['id']}/review", json={"fields": {"bogus": "1"}})
    assert bad.status_code == 422


def test_dashboard_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200 and "ledgerlens" in response.text
