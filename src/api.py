"""FastAPI service for the fine-tuned extractor.

The input is receipt *text*, not an image. Project 1's service took an upload and ran a
vision model over it; Qwen2.5-7B-Instruct has no vision tower, so OCR happens upstream and
this service receives words. That is a real consequence of choosing a text-only LLM and it
belongs in the API surface rather than being hidden behind a converter.

MODEL_BACKEND selects where generation happens:

    stub    canned output, no weights - so the container starts, serves and is CI-tested
            with no GPU present, exactly as Project 1's did
    hf      transformers, adapter loaded from MODEL_ADAPTER
    vllm    vLLM, which is what the benchmark measures

Endpoints:

    GET  /health
    GET  /metrics                measured results, or an honest "not yet"
    POST /extract                words in, structured fields out
    POST /analyze                extract, then run the deterministic checks
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .convert import Extraction, parse_prediction, to_receipt
from .schemas import ExtractedReceipt, Finding
from .sft_data import SYSTEM_PROMPT, build_prompt
from .tools import (
    check_line_item_arithmetic,
    check_line_items_sum_to_subtotal,
    check_total_reconciles,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

BACKEND = os.getenv("MODEL_BACKEND", "stub")
ADAPTER = os.getenv("MODEL_ADAPTER", "outputs/qwen-cord-lora")

# Receipt text is untrusted input. Cap it before the tokenizer sees it: an unbounded
# prompt is a way to make one request occupy the GPU indefinitely.
MAX_WORDS = 4000

app = FastAPI(
    title="Receipt extraction: QLoRA Qwen2.5-7B",
    description=(
        "Structured field extraction from receipt text, scored against the same "
        "entity-level metric as the LayoutLMv3 encoder it is compared with."
    ),
    version="0.1.0",
)


class ExtractRequest(BaseModel):
    doc_id: str = "upload"
    words: list[str] = Field(default_factory=list)
    text: str | None = None

    def word_list(self) -> list[str]:
        words = self.words or (self.text or "").split()
        if not words:
            raise HTTPException(400, "Provide either `words` or `text`.")
        if len(words) > MAX_WORDS:
            raise HTTPException(413, f"Receipt exceeds {MAX_WORDS} words.")
        return words


class AnalysisResponse(BaseModel):
    doc_id: str
    extraction: ExtractedReceipt
    fields: Extraction
    findings: list[Finding]


@lru_cache(maxsize=1)
def _generate():
    """Load the backend once, on first use rather than at import."""
    if BACKEND == "stub":
        canned = json.dumps(
            {"fields": [{"type": "total.total_price", "text": "TOTAL 31.000"}]}
        )
        return lambda prompts: [(canned, False) for _ in prompts]

    from .eval_llm import hf_backend, vllm_backend

    if BACKEND == "vllm":
        return vllm_backend(ADAPTER)
    if BACKEND == "hf":
        return hf_backend(ADAPTER)
    raise RuntimeError(f"Unknown MODEL_BACKEND {BACKEND!r}; expected stub, hf or vllm.")


def _extract_fields(words: list[str]) -> Extraction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(words)},
    ]
    raw, _ = _generate()([messages])[0]
    extraction = parse_prediction(raw)
    if extraction is None:
        # The same rule the evaluation uses: unparseable is a zero, not an exception and
        # not a silently repaired guess.
        raise HTTPException(502, "Model output was not valid JSON in the field contract.")
    return extraction


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_backend": BACKEND}


@app.get("/metrics")
def metrics() -> dict:
    """Serve the measured results, or say plainly that none exist yet."""
    available = {
        path.stem.removesuffix("_metrics"): json.loads(path.read_text())
        for path in sorted(ARTIFACTS.glob("*_metrics.json"))
    }
    project1 = ARTIFACTS / "project1" / "metrics.json"
    if project1.exists():
        available["layoutlmv3_project1"] = json.loads(project1.read_text())
    if not available:
        return {"available": False, "detail": "No evaluation run has completed yet."}
    return {"available": True, "results": available}


@app.post("/extract", response_model=Extraction)
def extract_endpoint(request: ExtractRequest) -> Extraction:
    return _extract_fields(request.word_list())


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_endpoint(request: ExtractRequest) -> AnalysisResponse:
    """Extract, then let Python decide whether the numbers reconcile.

    The findings are computed by `tools.py`, never generated by the model - the same
    division of labour as Project 1. The model reads; code does arithmetic.
    """
    fields = _extract_fields(request.word_list())
    receipt = to_receipt(request.doc_id, fields.fields)
    findings = [
        check_line_items_sum_to_subtotal(receipt.line_items, receipt.subtotal),
        check_total_reconciles(
            receipt.subtotal,
            receipt.tax,
            receipt.service_charge,
            receipt.discount,
            receipt.total,
        ),
        check_line_item_arithmetic(receipt.line_items),
    ]
    return AnalysisResponse(
        doc_id=request.doc_id, extraction=receipt, fields=fields, findings=findings
    )
