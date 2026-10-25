# Architecture

ledgerlens is a single Python package with four layers that only talk through typed
records (`ledgerlens.schemas`). Each layer can be exercised on its own, which is how the
tests and the benchmark are organised.

```
                 +-----------------------------+
                 |  generator                  |  seeded content, templates, fonts,
                 |  content -> render -> degrade|  degradation; ground truth per page
                 +--------------+--------------+
                                | PNG + GroundTruth (JSON)
                                v
+---------------------------------------------------------------------------+
| pipeline                                                                  |
|  preprocess ----> ocr (atlas | tesseract) ----> layout ----> extract      |
|  colour suppression,   segmentation + glyph      rows, fuzzy   fields,    |
|  deskew, Sauvola       matching, Viterbi,        labels,       items,     |
|                        calibration               tables        rows       |
|                                                      |                    |
|                                     validate <-------+------> confidence  |
|                                     rules (arithmetic, checksums, dates)  |
|                                                      |                    |
|                                              ExtractionResult + Decision  |
+---------------------------------------------------------------------------+
            |                                        |
            v                                        v
   +------------------+                    +---------------------+
   | bench            |                    | api                 |
   | splits, metrics, |                    | FastAPI, SQLite,    |
   | threshold, gate  |                    | queue, metrics, UI  |
   +------------------+                    +---------------------+
```

## Modules

| Module | Responsibility | Why it is shaped this way |
|---|---|---|
| `schemas.py` | Pydantic records: `DocumentFields`, `LineItem`, `Transaction`, `GroundTruth`, `ExtractionResult`, `RuleResult` | One vocabulary for gold and predictions; money is `Decimal`, dates are `date` |
| `canonical.py` | Canonical string forms for comparison | A match must mean the same information, not the same formatting |
| `generator/content.py` | Seeded invoices, receipts, statements with exact arithmetic | Distributions are chosen to look like paperwork (Poisson line counts, log-normal prices, running balances) |
| `generator/templates_*.py`, `canvas.py`, `spec.py`, `fonts.py` | Ten layouts, a recording canvas, per-document presentation choices, the seen/unseen font split | Word boxes come from the same font metrics Pillow rasterises with |
| `generator/degrade.py` | Skew, blur, motion blur, sensor noise, dust, vignetting, stamps, JPEG | Each step mimics one capture defect and is recorded per page |
| `generator/dataset.py` | Splits on disk with a manifest and fingerprint | Page `i` is identical whatever the split size; stale splits are detected |
| `pipeline/preprocess.py` | Colour-ink suppression, skew estimate by projection variance, noise-adaptive smoothing, Sauvola threshold, impulse cleanup | Order undoes capture defects one at a time |
| `pipeline/segment.py` | Components to glyphs, lines (vertical overlap plus horizontal proximity), block-local segments, words (pitch lattice for monospace, gap ratio otherwise) | Multi-column headers and mixed sizes on one row stay separate |
| `pipeline/ocr/atlas.py` | The shipped OCR: templates rendered from the seen fonts, intensity-normalised grey matching, fused-glyph splitting, Viterbi over character classes, logistic calibration | No Tesseract binary on the build machine; see README for the implication |
| `pipeline/ocr/tesseract.py`, `factory.py` | Optional Tesseract adapter and engine selection | Same `OCRPage` contract either way |
| `pipeline/layout.py` | Rows, fuzzy labels, values to the right or below, table headers to columns | Every extracted value is explainable by the label it sat next to |
| `pipeline/extract.py` | Document type, header fields, currency vote, line items, receipt items, transactions, arithmetic recovery of a missing amount | Anchored searches with per-type lexicons and arithmetic fallbacks |
| `pipeline/validate.py` | Totals, item sums, item arithmetic, dates, periods, IBAN and GSTIN checksums, balance chain, tax rate, currency | Cheap truth; failed hard rules force review |
| `pipeline/confidence.py` | Field confidence adjusted by rules, expected-field and labelled-field penalties, document confidence as a product, accept or review | A threshold on the product trades automation against error rate directly |
| `pipeline/calibration.py` | Per-field logistic calibration of confidences, fitted on the calibration split, saved as JSON | Raw confidences rank values within a field but are not comparable across fields |
| `pipeline/run.py` | `Pipeline.process(image)` | One call from pixels to a decision, with an optional LLM second reader |
| `llm/anthropic_adapter.py` | Optional Claude reviewer for low-confidence fields | Behind `LEDGERLENS_LLM=1`; rules still decide |
| `bench/` | Splits (parallel generation), metrics, runner (process pool, one OCR pass per page), calibration fit and rescoring, threshold selection, report, README block writer, gate | Numbers in the README come only from here |
| `api/` | FastAPI app, SQLite storage, review queue, Prometheus metrics, JSON logs, API-key auth, static dashboard | Low-confidence documents land in a queue; corrections are measured |
| `cli.py` | `generate`, `extract`, `bench`, `serve`, `atlas` | The same code paths the Makefile uses |

## Data flow for one page

1. `preprocess` converts RGB to grey with coloured pixels taking their brightest
   channel (stamps vanish, black text under them survives), estimates skew on a
   downsampled ink mask, rotates, smooths in proportion to measured noise, thresholds
   with Sauvola and removes isolated pixels.
2. `segment` labels connected components, separates ruling lines, groups glyphs into
   lines by vertical overlap and horizontal proximity, merges i-dots and colon halves,
   cuts rows into block-local segments, estimates baseline and cap height per segment
   and splits words.
3. The atlas engine vectorises each glyph (intensity-normalised grey crop, square
   padded, 20x20, blurred, unit norm) plus three geometric features, scores it against
   all 14,237 templates in one matrix product, reduces to the best score per character,
   splits fused blobs at the cut whose weaker half scores best (searched in one batch),
   smooths character classes with a Viterbi pass and converts scores to probabilities
   with the fitted logistic.
4. `layout.build_rows` merges segments into page-wide rows. `extract` classifies the
   document from vocabulary and paper shape, reads identifiers and votes on the
   currency (codes and labels count double, symbols once), takes the date order from
   that currency, then runs the anchored searches for each field, the table parser for
   line items or transactions (amounts are placed by their right edge), or the
   till-pattern parser for receipts. If exactly one of subtotal, tax and total is
   missing, the printed amount that satisfies the sum fills it. Optional fields whose
   label is printed but whose value was not read are flagged.
5. `validate.run_rules` checks arithmetic, checksums, dates and statement periods.
   `confidence.apply_rules` moves field confidences, the calibrator maps them to
   probabilities, `document_confidence` multiplies them with penalties for missing and
   flagged fields, and `decide` routes the page.

## Where state lives

- `data/atlas/atlas-<fingerprint>.npz`: the rendered template atlas and its calibration.
  Rebuilt automatically when fonts, charset or atlas version change.
- `data/synthetic/<split>/`: generated pages, ground truth and a manifest.
- `data/ledgerlens.db`: API storage (documents and reviews).
- `benchmarks/results.json`, `benchmarks/RESULTS.md`, `benchmarks/documents.jsonl`: the
  latest measurement, its tables, and one line per test document.
- `benchmarks/calibration.json`, `benchmarks/threshold.json`: the field calibration and
  threshold the API and CLI read at start-up (identity and 0.5 when absent).
- `benchmarks/thresholds.json`: the limits the benchmark gate enforces.

## Extending

- A new template is a function `(canvas, content, spec, rng) -> None` registered in
  `generator/templates.py`; the benchmark reports it as its own row.
- A new field is a `DocumentFields` attribute plus an extractor and, if it can be
  checked, a rule.
- A new OCR engine implements `recognize(gray, ink) -> OCRPage` and is registered in
  `pipeline/ocr/factory.py`.
