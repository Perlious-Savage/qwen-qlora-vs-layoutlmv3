"""API tests against the stub backend.

The stub exists so the container can be started and exercised with no GPU and no weights,
which is the only way this gets tested on every push rather than only on Colab.
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

RECEIPT = {
    "doc_id": "doc-1",
    "text": "ES TEH 8.000 SUBTOTAL 31.000 TOTAL 31.000",
}


def test_health_reports_the_backend():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_backend": "stub"}


def test_metrics_reports_project1_even_before_this_project_has_results():
    body = client.get("/metrics").json()
    assert body["available"] is True
    assert body["results"]["layoutlmv3_project1"]["test_f1"] == 0.9484066767830045


def test_extract_returns_fields_in_the_contract():
    body = client.post("/extract", json=RECEIPT).json()
    assert body["fields"][0]["type"] == "total.total_price"


def test_analyze_runs_the_deterministic_checks():
    body = client.post("/analyze", json=RECEIPT).json()
    assert body["extraction"]["total"] == "31000"
    assert [f["check_id"] for f in body["findings"]] == [
        "line_items_sum_to_subtotal",
        "total_reconciles",
        "line_item_arithmetic",
    ]


def test_empty_input_is_rejected_not_sent_to_the_model():
    assert client.post("/extract", json={"doc_id": "d", "text": ""}).status_code == 400


def test_oversized_input_is_rejected_before_tokenisation():
    huge = {"doc_id": "d", "words": ["X"] * 4001}
    assert client.post("/extract", json=huge).status_code == 413
