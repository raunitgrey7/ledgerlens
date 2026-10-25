"""Ask Claude to re-read the fields the deterministic pipeline is unsure about.

This is an enhancement behind ``LEDGERLENS_LLM=1``: the page image and the current
values go to the model, which answers with a JSON object of corrected values. The
adapter only touches fields that were low in confidence, canonicalises what comes
back, and leaves the final decision to the same validation rules as before, so the
LLM can never bypass a failed checksum or reconciliation. Any API failure or refusal
degrades to "no suggestions".
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Protocol

from PIL import Image

from ledgerlens.canonical import canonical_value
from ledgerlens.pipeline.parse import parse_date_text, parse_money
from ledgerlens.schemas import DATE_FIELDS, FIELD_NAMES, MONEY_FIELDS, ExtractedField, ExtractionResult

logger = logging.getLogger("ledgerlens.llm")
DEFAULT_MODEL = "claude-opus-5-5"
LOW_CONFIDENCE = 0.9
LLM_CONFIDENCE = 0.6
SYSTEM_PROMPT = (
    "You verify fields that an OCR pipeline extracted from one financial document image "
    "(an invoice, a receipt or a bank statement). Read the image carefully. Reply with a single JSON object "
    "and nothing else. Keys are the field names you were asked about; values are strings or null when the "
    "field is not printed on the page. Formats: money as a plain decimal with two places and no separators "
    "(1234.50); dates as ISO 8601 (2026-03-14); IBAN and GSTIN without spaces; names exactly as printed."
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class MessageClient(Protocol):
    """The slice of the Anthropic client the adapter uses, so tests can substitute a fake."""

    def create(self, **kwargs: Any) -> Any: ...


class ClaudeReviewer:
    """Second reader for uncertain fields."""

    def __init__(self, client: MessageClient | None = None, model: str = DEFAULT_MODEL, effort: str = "low") -> None:
        self.model = model
        self.effort = effort
        self._client = client

    @property
    def client(self) -> MessageClient:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic().beta.messages
        return self._client

    def review(self, image: Image.Image, result: ExtractionResult, fields: list[str]) -> dict[str, str | None]:
        """Canonical values Claude reads for ``fields``; empty on refusal, error or unparsable output."""
        if not fields:
            return {}
        try:
            response = self.client.create(
                model=self.model,
                max_tokens=4096,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [_image_block(image), {"type": "text", "text": _prompt(result, fields)}],
                    }
                ],
            )
        except Exception as exc:
            logger.warning("llm review failed", extra={"data": {"error": repr(exc)}})
            return {}
        if getattr(response, "stop_reason", None) == "refusal":
            return {}
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        )
        return parse_suggestions(text, fields)


def _image_block(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _prompt(result: ExtractionResult, fields: list[str]) -> str:
    current = {name: result.field_map().get(name) for name in fields}
    return (
        f"Document type: {result.doc_type.value}. Current extraction for the fields to verify: "
        f"{json.dumps(current)}. Return the correct value for each of these fields: {', '.join(fields)}."
    )


def parse_suggestions(text: str, fields: list[str]) -> dict[str, str | None]:
    """Extract the JSON object from the reply and keep only known, requested fields in canonical form."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    suggestions: dict[str, str | None] = {}
    for name in fields:
        if name not in payload or name not in FIELD_NAMES:
            continue
        value = payload[name]
        coerced = _coerce(name, value)
        if value in (None, "") or coerced is not None:
            suggestions[name] = coerced
    return suggestions


def _coerce(name: str, value: object) -> str | None:
    """Parse the model's string with the same tolerant parsers the pipeline uses, then canonicalise."""
    if value in (None, ""):
        return None
    text = str(value)
    if name in MONEY_FIELDS:
        parsed = parse_money(text)
        return canonical_value(name, parsed) if parsed is not None else None
    if name in DATE_FIELDS:
        parsed_date = parse_date_text(text)
        return canonical_value(name, parsed_date) if parsed_date is not None else None
    try:
        return canonical_value(name, text)
    except (ValueError, ArithmeticError, TypeError):
        return None


def low_confidence_fields(result: ExtractionResult, cutoff: float = LOW_CONFIDENCE) -> list[str]:
    """Fields worth a second look: present but uncertain."""
    return [field.name for field in result.fields if field.confidence < cutoff]


def apply_suggestions(
    fields: dict[str, ExtractedField], suggestions: dict[str, str | None]
) -> dict[str, ExtractedField]:
    """Agreement raises confidence; disagreement replaces the value at a fixed, modest confidence."""
    merged = {name: field.model_copy() for name, field in fields.items()}
    for name, value in suggestions.items():
        current = merged.get(name)
        if current is not None and current.value == value:
            current.confidence = round(1.0 - (1.0 - current.confidence) * 0.5, 4)
            current.evidence = f"{current.evidence}; confirmed by llm"
        elif value is not None:
            merged[name] = ExtractedField(name=name, value=value, confidence=LLM_CONFIDENCE, evidence="llm review")
    return merged
