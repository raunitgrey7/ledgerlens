"""The end-to-end pipeline: image in, validated and scored record out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from ledgerlens.pipeline.calibration import FieldCalibrator
from ledgerlens.pipeline.confidence import apply_rules, decide, document_confidence, missing_fields
from ledgerlens.pipeline.extract import Extraction, extract
from ledgerlens.pipeline.layout import build_rows
from ledgerlens.pipeline.ocr.base import OCREngine, OCRPage
from ledgerlens.pipeline.preprocess import preprocess
from ledgerlens.pipeline.validate import run_rules
from ledgerlens.schemas import FIELD_NAMES, Decision, ExtractedField, ExtractionResult

DEFAULT_THRESHOLD = 0.5


class Reviewer(Protocol):
    """Optional second reader (see ``ledgerlens.llm``) consulted only for documents headed to review."""

    def review(self, image: Image.Image, result: ExtractionResult, fields: list[str]) -> dict[str, str | None]: ...


@dataclass
class Pipeline:
    """Holds the OCR engine, the acceptance threshold and an optional reviewer; stateless otherwise."""

    engine: OCREngine
    threshold: float = DEFAULT_THRESHOLD
    reviewer: Reviewer | None = None
    calibrator: FieldCalibrator = field(default_factory=FieldCalibrator)

    def process(self, image: Image.Image) -> ExtractionResult:
        """Image in, scored and routed record out."""
        return self.process_with_page(image)[0]

    def process_with_page(self, image: Image.Image) -> tuple[ExtractionResult, OCRPage]:
        """Like ``process`` but also returns the OCR page, for metrics that score the OCR itself."""
        started = time.perf_counter()
        prepared = preprocess(image)
        page = self.engine.recognize(prepared.gray, prepared.ink)
        extraction = extract(page, build_rows(page))
        result = self._score(extraction, extraction.fields, page, prepared.skew_deg, started)
        if self.reviewer is not None and result.decision is Decision.REVIEW:
            from ledgerlens.llm.anthropic_adapter import apply_suggestions, low_confidence_fields

            suggestions = self.reviewer.review(image, result, low_confidence_fields(result))
            if suggestions:
                result = self._score(
                    extraction, apply_suggestions(extraction.fields, suggestions), page, prepared.skew_deg, started
                )
        return result, page

    def _score(
        self, extraction: Extraction, fields: dict[str, ExtractedField], page: OCRPage, skew: float, started: float
    ) -> ExtractionResult:
        rules = run_rules(extraction.doc_type, fields, extraction.line_items, extraction.transactions)
        scored = self.calibrator.apply(apply_rules(fields, rules))
        confidence = document_confidence(
            extraction.doc_type, scored, extraction.doc_type_confidence, extraction.flagged
        )
        ordered = [scored[name] for name in FIELD_NAMES if name in scored]
        return ExtractionResult(
            doc_type=extraction.doc_type,
            doc_type_confidence=extraction.doc_type_confidence,
            fields=ordered,
            line_items=extraction.line_items,
            transactions=extraction.transactions,
            rules=rules,
            document_confidence=confidence,
            missing_fields=missing_fields(extraction.doc_type, scored, extraction.flagged),
            decision=decide(confidence, rules, self.threshold),
            ocr_engine=page.engine,
            skew_deg=skew,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
            page_text=page.text,
        )
