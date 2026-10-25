# REPORT

## What was built

ledgerlens is document intelligence for invoices, receipts and bank statements, built to
the Forge charter and the `ledgerlens` section of SPECS.md.

- **Synthetic generator** (`src/ledgerlens/generator/`): seeded content with exact
  arithmetic (Poisson line counts, log-normal prices, running balances, valid IBAN
  mod-97 and GSTIN check characters), ten templates (five invoice, three receipt, two
  statement), four seen font families and two unseen faces, four locales, and recorded
  degradations (skew, Gaussian and motion blur, sensor noise, dust, vignetting,
  coloured stamps, JPEG). Every page ships with fields, line items or transactions,
  degradation parameters and a box for every printed word.
- **Pipeline** (`src/ledgerlens/pipeline/`): colour-ink suppression, projection-profile
  deskew, noise-adaptive smoothing, Sauvola binarisation; segmentation into lines,
  block-local segments and words; OCR; layout rows, fuzzy labels and tables; field,
  line-item, receipt-item and transaction extraction; validation rules (totals, item
  sums, item arithmetic, tax rate, dates, statement period and issue date, IBAN and
  GSTIN checksums, balance chain, currency); confidence with per-field calibration and
  missing-field penalties; accept or review.
- **OCR engine.** No `tesseract` binary exists on this Mac and `brew` is not installed,
  so nothing was installed. The package ships its own engine, a glyph atlas rendered
  from the four seen font families (14,237 templates after condensation), matched by
  intensity-normalised grey crops plus geometry, with a batched cut search for fused
  glyphs, Viterbi smoothing over character classes and a fitted logistic for character
  confidence. **Every benchmark number in this report and in README.md was produced by
  this atlas engine.** A Tesseract adapter is included and unit-tested with a fake
  module; it is selected automatically when a binary is present, and it was not
  measured here.
- **Benchmark** (`src/ledgerlens/bench/`): seeded calibration and test splits, exact
  and relaxed field and row metrics, line-level CER, per-field calibration fitted on the
  calibration split, threshold selection, effort model, latency, per-document output,
  README block writer, and a gate against `benchmarks/thresholds.json`.
- **API** (`src/ledgerlens/api/`): FastAPI with typed Pydantic schemas, `/health`,
  Prometheus `/metrics`, JSON logs with request ids, `X-API-Key` auth when
  `APP_API_KEY` is set, upload validation (type, size, dimensions), SQLite storage, a
  review queue, corrections with measured effort saved, `/v1/stats`, and a vanilla-JS
  dashboard at `/`.
- **Optional LLM step** (`src/ledgerlens/llm/`): a Claude second reader behind
  `LEDGERLENS_LLM=1` for low-confidence fields of documents already headed to review;
  validation rules still decide. Never used by tests or the benchmark.
- **Packaging**: Makefile (`setup`, `lint`, `test`, `bench`, `bench-gate`, `serve`),
  multi-stage Dockerfile with a non-root user, GitHub Actions CI with lint, tests and
  the benchmark gate. No `docker-compose.yml`, because there is one service.

## Environment

macOS 26.5.1 on Apple Silicon (arm64, 15 cores), Python 3.13.17 managed by uv 0.13.5,
Docker server 29.8.2. All runs below used the commands shown, from
`~/forge/ledgerlens`. "Clean shell" means `env -i HOME="$HOME"
PATH="$HOME/.local/bin:/usr/bin:/bin" make <target>`.

## Commands and outputs

### make lint (clean shell)

```
uv run ruff check src tests
All checks passed!
uv run ruff format --check src tests
73 files already formatted
```

### make test (clean shell)

```
uv run pytest -q
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 28.63s
```

The 116 tests cover unit behaviour of every module, property-based round trips
(Hypothesis: money and date formatting, identifier generation, canonical forms),
integration through the pipeline and the API, adversarial and robustness cases
(`tests/test_adversarial.py`: an injected "TOTAL 999,999.00" in a description, stamps
over totals, 4 degree skew, blank and noise pages) and determinism
(`tests/test_determinism.py`: identical pages, truth, split fingerprints and pipeline
output for the same seed). `tests/test_readme_numbers.py` fails if the README results
block differs from `benchmarks/results.json`.

### make bench (clean shell): the published run

Run at commit fba3201 (recorded in `benchmarks/results.json`), 200 calibration pages
(seed 101) and 400 test pages (seed 303), four worker processes, 283 s.

```
uv run python -m ledgerlens.bench --jobs 4
| Metric | Value |
|---|---|
| Documents scored (test split) | 400 |
| Document type accuracy | 97.0% |
| Field precision / recall / F1 (micro, exact match) | 81.9% / 51.9% / 63.6% |
| Line-item F1 (exact description, quantity, amount) | 31.5% |
| Transaction F1 (exact row) | 31.4% |
| Line-item F1 / transaction F1, amounts exact and description similarity at least 0.8 | 49.8% / 38.6% |
| OCR character error rate, seen fonts / unseen fonts | 23.9% / 31.8% |
| Fully correct records (all fields and rows) | 6.8% |
| Acceptance threshold (lowest with 95% header accuracy on the calibration split) | 1.0 |
| Accepted without review | 0.0% |
| Accepted documents with every header field exact | 0.0% |
| Accepted documents also exact on every line item and transaction | 0.0% |
| Human effort saved (cost model in docs/EVALUATION.md) | 41.5% |
| Latency per page, mean / p50 / p95 (ms) | 1522.2 / 1119.7 / 3766.8 |
wrote benchmarks/results.json (283s)
uv run python -m ledgerlens.bench.readme
updated README.md from benchmarks/results.json
```

Per-field, per-template, per-severity, per-font tables and the acceptance curve are in
README.md and `benchmarks/RESULTS.md`, written by the same command. Selected values
from `benchmarks/results.json`:

| Group | Field F1 | OCR CER |
|---|---|---|
| clean pages (100) | 91.0% | 3.5% |
| light pages (146) | 72.2% | 13.2% |
| heavy pages (154) | 26.1% | 51.4% |
| seen fonts (339) | 68.0% | 23.9% |
| unseen fonts (61) | 36.0% | 31.8% |

Acceptance curve, documents accepted (share) and header accuracy of the accepted:

| Threshold | Calibration split (200) | Test split (400) |
|---|---|---|
| 0.3 | 25 (12.5%), 84.0% | 58 (14.5%), 89.7% |
| 0.5 | 19 (9.5%), 84.2% | 47 (11.8%), 93.6% |
| 0.7 | 6 (3.0%), 66.7% | 17 (4.2%), 100.0% |
| 0.8 | 6 (3.0%), 66.7% | 14 (3.5%), 100.0% |
| 0.9 | 1 (0.5%), 100.0% | 2 (0.5%), 100.0% |

The threshold rule (95% header accuracy with at least five accepted documents, both
fixed before this run) found no qualifying threshold on the calibration split, so the
shipped threshold is 1.0 and the test split shows 0% acceptance. The two highest-scoring
calibration errors are vendor names: "NORDKAUF MARKT" read as "NORDKAUFMARKT" at
document confidence 0.861 and "URBAN GROCER" read as "URBAN GROEER" at 0.842. The
calibration and the threshold were fitted on the calibration split only; the test-split
curve is reported, not used.

### CI configuration with the gate (the command CI runs, with more workers)

```
uv run python -m ledgerlens.bench --test-size 120 --cal-size 200 --jobs 4 --data <scratch> --out <scratch> --gate
```

Exit status 0 (no gate failures), 141 s. Test split of 120 pages: document type accuracy
95.8%, field F1 62.5% (precision 81.8%, recall 50.5%), exact line-item F1 30.8%, exact
transaction F1 34.1%, OCR CER 27.3% on seen fonts and 27.0% on unseen fonts, latency
mean / p50 / p95 1589.7 / 1222.3 / 3727.8 ms, 0 documents accepted. The limits in
`benchmarks/thresholds.json` sit below these values (field F1 at least 0.60, line items
0.27, transactions 0.28, type accuracy 0.93, seen CER at most 0.30, p95 at most
12,000 ms, wrongly accepted share at most 0.02); the full published run also passes
them.

### From scratch, in a fresh clone

`git clone ~/forge/ledgerlens <scratch>/clone`, then clean-shell `make setup` (25 s:
`uv sync --all-extras` plus building and calibrating the atlas), `make lint` (all checks
passed, 73 files formatted), `make test` (116 passed in 28.60s). That clone was made at
commit a7156ac, which exposed a reproducibility bug: the clone's atlas had the same
fingerprint as the long-lived checkout but a different character calibration, because
the fingerprint did not cover the recogniser source. Commit fba3201 fixed it. After
pulling fba3201 into the clone, the same command was run in both trees:

```
uv run python -m ledgerlens.bench --test-size 30 --cal-size 30 --jobs 4 --data <dir> --out <dir>
```

Result of comparing the two `results.json` files: split fingerprints equal, all results
equal apart from latency and run metadata, `calibration.json` files identical.

### Docker

`docker build -t ledgerlens:latest .` succeeded in 33 s (layers cached), image size
621 MB. Smoke test: `docker run -p 18080:8000 ledgerlens:latest`, `GET /health` returned
`{"status":"ok","version":"0.1.0","ocr_engine":"atlas","threshold":1.0}`, uploading
`data/synthetic/test/test-00001.png` to `POST /v1/documents` returned an invoice routed
to review with 9 fields and a pipeline latency of 2092.7 ms inside the container, the
process ran as user `app` (uid 10001), and the image's atlas fingerprint matched the
host's.

### make serve

Clean-shell `make serve PORT=18081`: `GET /health` returned status ok with threshold
1.0, `GET /` returned the dashboard (HTTP 200, 7,260 bytes), `GET /v1/queue` returned an
empty queue, and every request produced a JSON log line with a request id.

## How the result got here

The first complete benchmark showed that the document confidence did not separate good
records from bad ones: the documents it accepted had header errors. The causes, found
on calibration pages and development seeds, were fixed one by one:

- The document score ignored missing optional fields and treated a name confidence
  like an amount confidence. Fixed with expected-field and labelled-field penalties and
  a per-field calibration fitted on the calibration split.
- The fused-glyph splitter cut hollow letters in the middle ("02" became "I" and "J");
  replaced by a batched search for the cut whose weaker half scores best, with a guard
  that keeps a lowercase m whole.
- The extractor dropped "&" from names, vetoed the label "Invoice" as if it were
  "Tax Invoice", let a "Ship To" column leak into the customer, placed wide amounts in
  the description column, took the date order from a separate vote that one misread "$"
  could flip, and appended quantities to descriptions when the "Qty" header was misread.
- The generator sometimes printed no currency at all while the ground truth required
  one; every page now states its currency.
- The atlas cache key ignored the recogniser source (described above).

The test seed was changed from 202 to 303 for the final runs, because seed 202 had been
used during development. One run on seed 303 before the published one was looked at in
aggregate; its per-template table led to the Qty-header fix, which was diagnosed on
calibration pages. The acceptance target was not changed after seeing test results.

## Known gaps

- **No automatic acceptance.** At the 95% bar the calibration split supports no
  threshold, so the shipped configuration sends every document to review. The test
  split curve suggests a threshold near 0.7 would work, but choosing it from the test
  split would not be an honest measurement; a larger calibration split is the right fix.
- **Recall.** Micro recall is 51.9% against 81.9% precision. Most misses are fields not
  found at all, usually because the OCR misread the label.
- **Heavy degradation and unseen fonts.** 26.1% field F1 on heavy pages and 36.0% on
  unseen fonts. The atlas OCR has no learned model and no language model.
- **Names.** Vendor and customer names (46.4% and 36.5% F1) have no checksum and cause
  most confidently wrong records.
- **Strict rows.** Exact line-item F1 is 31.5%; with relaxed descriptions it is 49.8%.
- **Tesseract not measured.** The adapter exists and is unit-tested only.
- **CI not run on GitHub.** Nothing was pushed, so the workflow has only been exercised
  locally through the same commands. Font rasterisation on Linux may differ slightly
  from macOS; the gate limits leave a margin for that, but this was not measured.
- **LLM reviewer not run against the API.** It is tested with a fake client only.
- **Latency.** 1.5 s mean and 3.8 s p95 per page on this machine; heavy pages are the
  slow ones. CI runner latency was not measured.
- **Synthetic data only.** No public scanned dataset was evaluated.
