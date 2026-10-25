"""HTTP routes: ingest documents, read results, work the review queue, report stats."""

from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from ledgerlens import __version__
from ledgerlens.api.schemas import (
    DocumentResponse,
    HealthResponse,
    QueueItem,
    QueuePage,
    ReviewRequest,
    ReviewResponse,
    StatsResponse,
)
from ledgerlens.api.storage import StoredDocument
from ledgerlens.bench.metrics import EffortModel
from ledgerlens.schemas import FIELD_NAMES, ExtractionResult

router = APIRouter()
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "TIFF", "BMP", "WEBP"})
EFFORT = EffortModel()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        ocr_engine=request.app.state.pipeline.engine.name,
        threshold=settings.threshold,
    )


@router.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics(request: Request) -> Response:
    request.app.state.metrics.queue_depth.set(request.app.state.storage.count("pending"))
    return Response(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


def _decode_upload(request: Request, upload: UploadFile, payload: bytes) -> Image.Image:
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > request.app.state.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload too large")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=415, detail="file is not a readable image") from exc
    if (image.format or "").upper() not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail=f"unsupported image format {image.format}")
    if image.width < 64 or image.height < 64 or image.width * image.height > 40_000_000:
        raise HTTPException(status_code=422, detail="image dimensions out of range")
    return image.convert("RGB")


@router.post("/v1/documents", response_model=DocumentResponse, status_code=201, tags=["documents"])
async def ingest(request: Request, file: UploadFile = File(...)) -> DocumentResponse:
    """Run the pipeline on an uploaded page; low-confidence results land in the queue."""
    payload = await file.read()
    image = _decode_upload(request, file, payload)
    result: ExtractionResult = request.app.state.pipeline.process(image)
    doc_id = uuid.uuid4().hex[:12]
    stored = request.app.state.storage.insert_document(doc_id, file.filename, result.model_dump(mode="json"))
    metrics = request.app.state.metrics
    metrics.documents.labels(result.doc_type.value, result.decision.value).inc()
    metrics.pipeline_latency.observe(result.latency_ms / 1000.0)
    request.app.state.logger.info(
        "document processed",
        extra={
            "data": {
                "document_id": doc_id,
                "doc_type": result.doc_type.value,
                "decision": result.decision.value,
                "confidence": result.document_confidence,
                "latency_ms": result.latency_ms,
            }
        },
    )
    return _document_response(stored)


def _document_response(stored: StoredDocument) -> DocumentResponse:
    return DocumentResponse(
        id=stored.id,
        created_at=stored.created_at,
        filename=stored.filename,
        status=stored.status,
        result=ExtractionResult.model_validate(stored.result),
    )


@router.get("/v1/documents/{doc_id}", response_model=DocumentResponse, tags=["documents"])
def get_document(request: Request, doc_id: str) -> DocumentResponse:
    stored = request.app.state.storage.get_document(doc_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _document_response(stored)


@router.get("/v1/queue", response_model=QueuePage, tags=["review"])
def queue(
    request: Request, limit: int = Query(default=20, ge=1, le=200), offset: int = Query(default=0, ge=0)
) -> QueuePage:
    """Documents waiting for a human, oldest first within the page."""
    storage = request.app.state.storage
    items = []
    for stored in storage.list_documents("pending", limit, offset):
        result = ExtractionResult.model_validate(stored.result)
        items.append(
            QueueItem(
                id=stored.id,
                created_at=stored.created_at,
                filename=stored.filename,
                doc_type=result.doc_type.value,
                confidence=result.document_confidence,
                failed_rules=[r.name for r in result.rules if r.passed is False],
                fields=result.field_map(),
            )
        )
    return QueuePage(items=items, pending=storage.count("pending"), limit=limit, offset=offset)


@router.post("/v1/queue/{doc_id}/review", response_model=ReviewResponse, tags=["review"])
def review(request: Request, doc_id: str, body: ReviewRequest) -> ReviewResponse:
    """Store corrections and measure how much of the prefilled record survived."""
    storage = request.app.state.storage
    stored = storage.get_document(doc_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="document not found")
    unknown = sorted(set(body.fields) - set(FIELD_NAMES))
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown fields: {', '.join(unknown)}")
    result = ExtractionResult.model_validate(stored.result)
    predicted = result.field_map()
    corrected = sum(1 for name, value in body.fields.items() if predicted.get(name) != value)
    fields_total = max(len(predicted), len({*predicted, *body.fields}))
    storage.record_review(
        doc_id,
        body.reviewer,
        fields_total,
        corrected,
        body.line_items_corrected,
        {"fields": body.fields, "note": body.note},
    )
    metrics = request.app.state.metrics
    metrics.corrections.labels("corrected").inc(corrected)
    metrics.corrections.labels("confirmed").inc(fields_total - corrected)
    saved = _effort_saved(fields_total, corrected, 0)
    return ReviewResponse(
        document_id=doc_id,
        fields_total=fields_total,
        fields_corrected=corrected,
        fields_confirmed=fields_total - corrected,
        effort_saved_fraction=saved,
    )


def _effort_saved(reviewed: int, corrected: int, auto_accepted: int) -> float:
    """1 minus assisted cost over manual cost under the benchmark's effort model."""
    manual = (reviewed + auto_accepted) * EFFORT.manual
    if manual == 0:
        return 0.0
    assisted = corrected * EFFORT.correct + (reviewed - corrected) * EFFORT.verify
    return round(1.0 - assisted / manual, 4)


@router.get("/v1/stats", response_model=StatsResponse, tags=["review"])
def stats(request: Request) -> StatsResponse:
    storage = request.app.state.storage
    totals = storage.review_totals()
    auto_fields = storage.accepted_field_count()
    return StatsResponse(
        documents=storage.count(),
        accepted=storage.count("accepted"),
        pending=storage.count("pending"),
        reviewed=storage.count("reviewed"),
        reviews=totals["reviews"],
        fields_reviewed=totals["fields_total"],
        fields_corrected=totals["fields_corrected"],
        fields_auto_accepted=auto_fields,
        effort_saved_fraction=_effort_saved(totals["fields_total"], totals["fields_corrected"], auto_fields),
        effort_model=EFFORT.as_dict(),
        threshold=request.app.state.settings.threshold,
        ocr_engine=request.app.state.pipeline.engine.name,
    )
