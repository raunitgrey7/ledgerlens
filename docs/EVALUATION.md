# Evaluation

Every number in README.md and REPORT.md is produced by

```
make bench
# which runs
uv run python -m ledgerlens.bench --jobs 4
uv run python -m ledgerlens.bench.readme
```

The first command writes `benchmarks/results.json`, `benchmarks/RESULTS.md`,
`benchmarks/documents.jsonl`, `benchmarks/calibration.json` and
`benchmarks/threshold.json`; the second copies the tables into README.md between two
markers, and `tests/test_readme_numbers.py` fails if the README block and
`results.json` ever disagree. Accuracy numbers are deterministic for a given machine and
dependency set: splits are seeded, the OCR atlas is rendered from vendored fonts, and
the pipeline uses no randomness. Latency varies from run to run.

## Dataset

Two synthetic splits, generated and cached under `data/synthetic/`:

| Split | Pages | Seed | Used for |
|---|---|---|---|
| calibration | 200 | 101 | fitting the per-field confidence calibration and choosing the acceptance threshold |
| test | 400 | 303 | every reported metric |

**Seed policy.** During development the pipeline was debugged on pages from seeds 7,
11, 4242 and 202 and on the calibration split; seed 202 was the test seed until then and
is therefore no longer held out. Seed 303 was introduced for the final runs. One run on
it before the published one was looked at in aggregate only: its per-template table
showed exact line-item F1 near zero for three invoice templates, the cause (a misread
"Qty" header) was diagnosed on calibration pages and fixed, and the published run is the
second run on seed 303. No per-document output of the test split was used for any
decision, and the acceptance target was not changed after seeing it. Page `i` of a split
depends only on (seed, i), so a smaller split is a prefix of a larger one.

Page mix (per page, drawn with the seed): document type invoice 50% / receipt 30% /
bank statement 20%; severity clean 20% / light 40% / heavy 40%; font family drawn from
the four seen families 85% of the time and from the two unseen faces (Pillow's embedded
Aileron, DejaVu Serif Italic) 15% of the time; template uniform within the type. Ground
truth per page: header fields, line items or transactions, the exact degradation
parameters and a box for every printed word. Every page states its currency, either on
the amounts (symbol or ISO code) or in a currency line.

Degradation profiles (`generator/degrade.py`):

| Severity | Rotation | Gaussian blur sigma | Noise sigma | Dust | Vignetting | Stamp | JPEG |
|---|---|---|---|---|---|---|---|
| clean | 0 | 0 | 0 to 1.5 | 0 | 0 | no | no |
| light | up to 1.5 deg | 0.3 to 0.7 | 3 to 8 | up to 0.1% | 3 to 12% | 30% | q70 to 90 (50%) |
| heavy | up to 4 deg | 0.7 to 1.3, plus 3 to 5 px motion blur in 30% | 8 to 18 | 0.1 to 0.4% | 10 to 30% | 60% | q35 to 65 (80%) |

The dataset is synthetic. It has exact ground truth at any volume, but it only covers
the vocabulary, layouts and fonts the generator knows; see Limitations in README.md.

## Metrics

**Field precision, recall and F1.** For each document and each of the 15 header
fields, the gold value g and predicted value p are compared after canonicalisation
(`canonical.py`: cent-precision money, ISO dates, upper-case identifiers without
spaces, upper-case names with punctuation removed and single spaces). Counting follows
the SROIE convention:

| gold | prediction | counts as |
|---|---|---|
| g | p == g | true positive |
| g | p present, p != g | one false positive and one false negative |
| g | absent | false negative |
| absent | p present | false positive |
| absent | absent | ignored |

Micro precision = TP / (TP + FP), recall = TP / (TP + FN), over all fields and
documents. Per-field, per-template, per-severity and per-font tables use the same
counts restricted to the group.

**Line-item F1 and transaction F1 (exact).** Each gold row becomes the tuple
(canonical description, quantity, amount) for items or (date, canonical description,
debit, credit, balance) for transactions. Predicted rows are matched as a multiset
intersection; precision and recall follow.

**Relaxed row F1.** The same rows, but a prediction matches a gold row when every
numeric key is exact and the descriptions have a `difflib` similarity ratio of at least
0.8. Gold rows are matched greedily to the most similar unclaimed prediction. Both the
exact and the relaxed figure are reported; the relaxed one shows how much of the exact
gap is OCR noise in free text rather than wrong amounts.

**OCR character error rate.** For each ground-truth line, the predicted words whose
centres fall inside the line's box are joined with spaces and compared with the line
text by Levenshtein distance; CER = total edits / total gold characters. Reported for
seen and unseen fonts and per severity.

**Header-correct and fully correct.** A document is header-correct when none of its
header fields has a false positive or a false negative, and fully correct when in
addition every line item or transaction matches exactly.

**Confidence.** Field confidence starts as the OCR probability that the value is right
(a product over characters and words), times a label-match factor, and is moved by the
validation rules (a passed rule halves the remaining doubt, a failed one halves the
confidence). A per-field logistic regression on the logit of that confidence, fitted on
the calibration split with a ridge penalty towards the identity map, turns it into a
probability (`pipeline/calibration.py`; fields with fewer than 15 samples or a single
outcome keep the identity). The document confidence is the product of the calibrated
confidences of the fields present, times 0.2 for every field the document type should
carry but the pipeline did not find, 0.2 for every optional field whose label is
printed but whose value was not read (GSTIN, IBAN, a receipt number), and
min(1, document-type confidence + 0.3), where the type confidence is the winning share
of the vocabulary and paper-shape votes.

**Acceptance.** A document is accepted when its confidence is at or above the threshold
and no hard rule failed (totals reconciliation, item sum, IBAN or GSTIN checksum,
balance chain). The threshold is chosen on the calibration split, after the calibration
fit: sweeping a 0.01 grid down from 1.0, thresholds that accept fewer than five
documents are skipped, the sweep stops at the first threshold where fewer than 95% of
accepted documents are header-correct, and the lowest threshold before that is chosen.
When no threshold qualifies the result is 1.0, which accepts nothing. The 95% target and
the five-document minimum were set before the published run. The calibration and the
threshold are then frozen and applied to the test split. The report shows the acceptance
rate, the header and full-record accuracy of what was accepted, the number of documents
accepted with a header error, and the same figures at fixed thresholds (0.3 to 0.95) as
an operating curve.

**Human effort saved.** A keystroke-equivalent cost model: entering a field by hand
costs 1.0; in a queued document, confirming a correct prefilled field costs 0.2 and
fixing a wrong or missing one costs 1.0; an accepted document costs nothing. Effort
saved = 1 minus assisted cost over manual cost. The same model backs `/v1/stats` in the
API, where the counts come from real reviews instead of the ground truth. Wrong
acceptances are not charged by the model, which is why they are reported next to it.

**Latency.** Wall time of `Pipeline.process` per page (preprocessing, OCR, layout,
extraction, validation, confidence), mean, median, 95th percentile and maximum,
measured inside the benchmark workers (four processes on the build machine).

## Gate

`uv run python -m ledgerlens.bench --gate` compares the fresh results with
`benchmarks/thresholds.json` and exits non-zero on any violation. CI runs it with
`--test-size 120 --cal-size 200 --jobs 2`: the calibration split is the full one, so the
calibration and threshold are the same as in the published run, and the test pages are
the first 120 of the published test split. The limits sit a few points below what that
configuration measured locally (see REPORT.md), with a wide latency limit because CI
runners are slower:

| Check | Limit |
|---|---|
| field micro F1 | at least 0.60 |
| exact line-item F1 | at least 0.27 |
| exact transaction F1 | at least 0.28 |
| document type accuracy | at least 0.93 |
| OCR CER on seen fonts | at most 0.30 |
| latency p95 | at most 12,000 ms |
| acceptance rate | at least 0.0 |
| documents accepted with a header error, as a share of all documents | at most 0.02 |

## Reproducing

```
make setup
make bench            # full sizes, writes benchmarks/ and refreshes README.md
make bench-gate       # same, then checks benchmarks/thresholds.json
uv run python -m ledgerlens.bench --test-size 40 --cal-size 40 --jobs 2 --out /tmp/quick   # quick look
```

`--rebuild` regenerates the splits; otherwise the manifest fingerprint and generator
version are checked and cached pages are reused. `--data DIR` puts the splits under
`DIR/synthetic`, which is how the CI configuration was measured without overwriting the
published splits.
