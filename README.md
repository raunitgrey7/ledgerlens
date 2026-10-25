# ledgerlens

Document intelligence for finance: a seeded generator of invoices, receipts and bank
statements with exact ground truth, an offline OCR and extraction pipeline with
accounting validation rules and calibrated confidence, a benchmark that measures every
claim, and a review-queue API that records what humans had to fix.

Everything runs on a laptop CPU with no API key and no network. Python 3.13 via `uv`.

## What it does

1. **Generates documents you can score against.** Ten layouts (five invoice, three
   receipt, two bank statement templates), four seen font families plus two held-out
   faces, four locales (INR with GSTIN, GBP and EUR with IBAN, USD), three separator
   and seven date conventions. Each page is degraded with physically motivated
   defects (skew, defocus and motion blur, sensor noise, dust, vignetting, coloured
   stamps, JPEG) and shipped with its fields, line items, transactions and word boxes.
2. **Reads them back.** Colour-ink suppression, deskew, noise-adaptive binarisation,
   segmentation, a glyph-atlas OCR engine built from the generator's own fonts (or
   Tesseract when a binary is present), fuzzy label anchoring, table and till-pattern
   parsing, checksummed identifiers.
3. **Checks the record.** Totals must reconcile, items must sum, IBAN and GSTIN
   checksums must hold, balances must chain, dates must make sense. Rules move field
   confidences; the document confidence is the product; a threshold chosen on a
   calibration split decides between straight-through processing and human review.
4. **Measures it.** Field precision, recall and F1 per field, template, severity and
   font; line-item and transaction F1; OCR character error rate; the fraction of
   documents accepted without review and how many of those were fully correct; human
   effort saved under an explicit cost model; latency per page.
5. **Serves it.** FastAPI with typed schemas, `/health`, Prometheus `/metrics`, JSON
   logs with request ids, API-key auth, SQLite persistence, a review queue with
   corrections, and a one-page dashboard.

## Architecture

```
generator ──PNG + truth──▶ preprocess ─▶ OCR (atlas | tesseract) ─▶ layout ─▶ extract
                                                                               │
                             bench ◀── ExtractionResult ◀── confidence ◀── validate
                               │                                   │
                        results.json                      API: queue, corrections, stats
```

`docs/ARCHITECTURE.md` describes every module and the data flow for one page.

## Quickstart

```
cd ~/forge/ledgerlens
make setup        # uv sync and build the OCR glyph atlas (about 15 s)
make test         # pytest, about 2 minutes on a laptop
make bench        # generate 550 pages, score them, write benchmarks/results.json
make serve        # API and dashboard on http://localhost:8000
```

Then process a page:

```
uv run ledgerlens generate --size 3 --name demo
curl -F "file=@data/synthetic/demo/demo-00000.png" http://localhost:8000/v1/documents
```

Or from the command line without the server: `uv run ledgerlens extract data/synthetic/demo/demo-00000.png`.

## Measured results

<!-- results:start -->
Run `make bench` to produce this table; the numbers below are copied from `benchmarks/results.json`.
<!-- results:end -->

**OCR engine used for these numbers: the shipped glyph-atlas engine (`atlas`).** No
`tesseract` binary existed on the build machine and none could be installed without
sudo, so the fallback engine described in `docs/ARCHITECTURE.md` produced every figure
above. It is trained on the four seen font families the generator uses, which is why
the table separates seen from unseen fonts: the unseen rows are the honest measure of
generalisation, the seen rows are what a deployment with known fonts would get. A
Tesseract adapter is included and selected automatically when the binary is present
(`LEDGERLENS_OCR=tesseract` forces it); its numbers are not reported because they were
not measured here.

Definitions, the dataset mix, the acceptance threshold procedure and the effort cost
model are in `docs/EVALUATION.md`. `REPORT.md` records the exact commands and outputs.

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | liveness, version, engine, threshold |
| `GET /metrics` | Prometheus text format |
| `POST /v1/documents` | multipart image upload; returns the typed extraction and decision |
| `GET /v1/documents/{id}` | a stored result |
| `GET /v1/queue` | documents waiting for review, with failed rules and prefilled fields |
| `POST /v1/queue/{id}/review` | corrections; returns fields corrected, confirmed and effort saved |
| `GET /v1/stats` | counts and the measured effort saving across all reviews |
| `GET /` | dashboard (vanilla JS, no build step) |

Set `APP_API_KEY` to require `X-API-Key` on `/v1/*`. Every response carries
`X-Request-ID`, echoed from the request when provided.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LEDGERLENS_DATA_DIR` | `data` | atlas cache, synthetic splits, database |
| `LEDGERLENS_OCR` | `auto` | `atlas`, `tesseract`, or `auto` (Tesseract if installed) |
| `LEDGERLENS_THRESHOLD` | from `benchmarks/threshold.json`, else 0.5 | acceptance threshold |
| `LEDGERLENS_DB` | `data/ledgerlens.db` | SQLite path |
| `APP_API_KEY` | unset | API key required on `/v1/*` when set |
| `LEDGERLENS_LLM` | `0` | `1` enables the optional Claude second reader |
| `LEDGERLENS_LLM_MODEL` | `claude-opus-5-5` | model for the second reader (`claude-haiku-5-5` is the cheap option) |
| `LEDGERLENS_MAX_UPLOAD_BYTES` | 20 MiB | upload limit |

The optional LLM step (`uv sync --extra llm`, `LEDGERLENS_LLM=1`, credentials resolved
by the Anthropic SDK) is consulted only for documents already headed to review; it
re-reads low-confidence fields from the page image and the validation rules still make
the decision. Tests and the benchmark never use it.

## Docker

```
docker build -t ledgerlens .
docker run -p 8000:8000 ledgerlens
```

Multi-stage build, non-root user, atlas prebuilt into the image.

## Limitations

- The data is synthetic. Vocabulary, layouts, fonts and degradations are those of the
  generator; real scans have more of everything. The seen-font numbers measure the
  pipeline on documents whose fonts the OCR has templates for.
- The atlas OCR is a nearest-template recogniser. It degrades sharply on heavy
  blur and on italic or unseen faces, and it has no language model beyond character
  class smoothing, so a misread label can hide a field.
- Receipts with tax included in prices, multi-page documents, handwriting, tables with
  wrapped descriptions and discounts are not generated and not handled.
- Day-month order for ambiguous numeric dates is inferred from the currency; a GBP
  document printed in month-day order would be misread.
- Document confidence assumes field errors are independent; the calibration split
  corrects the threshold, not the shape of that assumption.

## Roadmap

- A learned character classifier (small CNN trained on the generator) to replace
  nearest-template matching on heavy pages.
- Line-level language model for label words.
- Multi-page statements and invoices with carried totals.
- Active learning from review corrections: queued documents with corrections become
  regression cases for the benchmark.

## Development

```
make lint     # ruff check and format check
make test     # pytest
make bench    # benchmark, writes benchmarks/
make bench-gate   # benchmark with thresholds from benchmarks/thresholds.json
```

Fonts in `src/ledgerlens/assets/fonts` are DejaVu (Bitstream Vera licence) and STIX
(SIL Open Font Licence); licences are included next to the files.
