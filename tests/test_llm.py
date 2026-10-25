import types

from PIL import Image

from ledgerlens.llm.anthropic_adapter import ClaudeReviewer, apply_suggestions, low_confidence_fields, parse_suggestions
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.schemas import Decision, DocType, ExtractedField, ExtractionResult


def _result(**confs: float) -> ExtractionResult:
    fields = [ExtractedField(name=name, value="1.00", confidence=conf) for name, conf in confs.items()]
    return ExtractionResult(
        doc_type=DocType.INVOICE,
        doc_type_confidence=0.9,
        fields=fields,
        document_confidence=0.5,
        decision=Decision.REVIEW,
        ocr_engine="atlas",
    )


class FakeMessages:
    def __init__(self, text: str | None = None, stop_reason: str = "end_turn", error: Exception | None = None) -> None:
        self.text, self.stop_reason, self.error, self.calls = text, stop_reason, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        block = types.SimpleNamespace(type="text", text=self.text or "")
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[block])


def test_reviewer_sends_image_and_parses_json() -> None:
    fake = FakeMessages('Here you go: {"total": "1,234.50", "tax": null, "bogus": "x"}')
    reviewer = ClaudeReviewer(client=fake, model="claude-opus-5-5")
    out = reviewer.review(Image.new("RGB", (100, 100), "white"), _result(total=0.5, tax=0.5), ["total", "tax"])
    assert out == {"total": "1234.50", "tax": None}
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["messages"][0]["content"][0]["type"] == "image"


def test_reviewer_degrades_on_refusal_error_and_garbage() -> None:
    image = Image.new("RGB", (50, 50), "white")
    assert (
        ClaudeReviewer(client=FakeMessages("{}", stop_reason="refusal")).review(image, _result(total=0.5), ["total"])
        == {}
    )
    assert (
        ClaudeReviewer(client=FakeMessages(error=RuntimeError("down"))).review(image, _result(total=0.5), ["total"])
        == {}
    )
    assert ClaudeReviewer(client=FakeMessages("no json here")).review(image, _result(total=0.5), ["total"]) == {}
    assert ClaudeReviewer(client=FakeMessages("{}")).review(image, _result(total=0.5), []) == {}
    assert parse_suggestions('{"total": "abc"}', ["total"]) == {}


def test_suggestions_adjust_fields() -> None:
    fields = {
        "total": ExtractedField(name="total", value="1.00", confidence=0.6),
        "tax": ExtractedField(name="tax", value="2.00", confidence=0.6),
    }
    merged = apply_suggestions(fields, {"total": "1.00", "tax": "3.00", "iban": "GB82WEST12345698765432"})
    assert merged["total"].confidence == 0.8 and "confirmed" in merged["total"].evidence
    assert merged["tax"].value == "3.00" and merged["tax"].confidence == 0.6
    assert merged["iban"].value == "GB82WEST12345698765432"
    assert low_confidence_fields(_result(total=0.5, tax=0.95)) == ["total"]


def test_pipeline_consults_reviewer_only_for_review_decisions(engine, page_by_type) -> None:
    calls = []

    class Spy:
        def review(self, image, result, fields):
            calls.append(fields)
            return {}

    page, _ = page_by_type[DocType.INVOICE]
    strict = Pipeline(engine, threshold=1.01, reviewer=Spy())
    result = strict.process(page)
    assert result.decision is Decision.REVIEW and len(calls) == 1
    lenient = Pipeline(engine, threshold=0.0, reviewer=Spy())
    lenient.process(page)
    assert len(calls) == 1
