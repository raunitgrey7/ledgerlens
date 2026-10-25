# Evaluation

Every number in README.md and REPORT.md is produced by

```
uv run python -m ledgerlens.bench --test-size 400 --cal-size 150 --jobs 4
```

which writes `benchmarks/results.json`, `benchmarks/RESULTS.md` and
`benchmarks/threshold.json`. The command is deterministic for a given machine and
dependency set: splits are seeded, the OCR atlas is rendered from fixed fonts, and the
pipeline uses no randomness.

## Dataset

Two synthetic splits, generated and cached under `data/synthetic/`:

| Split | Pages | Seed | Used for |
|---|---|---|---|
| calibration | 150 | 101 | choosing the acceptance threshold |
| test | 400 | 202 | every reported metric |

Page mix (per page, drawn with the seed): document type invoice 50% / receipt 30% /
bank statement 20%; severity clean 20% / light 40% / heavy 40%; font family drawn
from the four seen families 85% of the time and from the two unseen faces (Pillow's
embedded Aileron, DejaVu Serif Italic) 15% of the time; template uniform within the
type. Ground truth per page: header fields, line items or transactions, the exact
degradation parameters and a word-level box list.

Degradation profiles (`generator/degrade.py`):

| Severity | Rotation | Gaussian blur sigma | Noise sigma | Dust | Vignetting | Stamp | JPEG |
|---|---|---|---|---|---|---|---|
| clean | 0 | 0 | 0 to 1.5 | 0 | 0 | no | no |
| light | up to 1.5 deg | 0.3 to 0.7 | 3 to 8 | up to 0.1% | 3 to 12% | 30% | q70 to 90 (50%) |
| heavy | up to 4 deg | 0.7 to 1.3, plus 3 to 5 px motion blur in 30% | 8 to 18 | 0.1 to 0.4% | 10 to 30% | 60% | q35 to 65 (80%) |

The dataset is synthetic. It has known ground truth at any volume, but it only covers
the vocabulary, layouts and fonts the generator knows; see Limitations in README.md.

## Metrics

**Field precision, recall and F1.** For each document and each of the 15 header
fields, gold value g and predicted value p are compared after canonicalisation
(`canonical.py`: cent-precision money, ISO dates, upper-case identifiers without
spaces, upper-case names with single spaces). Counting follows the SROIE convention:

| gold | prediction | counts as |
|---|---|---|
| g, p == g | | true positive |
| g, p != g, p present | | one false positive and one false negative |
| g, p absent | | false negative |
| absent, p present | | false positive |
| absent, absent | | ignored |

Micro precision = TP / (TP + FP), recall = TP / (TP + FN), over all fields and
documents. Per-field, per-template, per-severity and per-font tables use the same
counts restricted to the group.

**Line-item F1 and transaction F1.** Each gold row becomes the tuple (canonical
description, quantity, amount) for items or (date, canonical description, debit,
credit, balance) for transactions. Predicted rows are matched as a multiset
intersection; precision and recall follow.

**OCR character error rate.** For each ground-truth line, the predicted words whose
centres fall inside the line's box are joined with spaces and compared with the
line text by Levenshtein distance; CER = total edits / total gold characters. Reported
for seen and unseen fonts and per severity.

**Fully correct record.** A document whose header fields have no false positive or
false negative and whose rows all match.

**Acceptance.** `document_confidence` is the product of the confidences of the
required fields (missing ones count 0.35) and of every optional field present, after
the rule adjustments. A document is accepted when the confidence is at or above the
threshold and no hard rule failed (totals reconciliation, item sum, IBAN or GSTIN
checksum, balance chain). The threshold is the lowest value on a 0.01 grid for which
the accepted records of the calibration split are fully correct at least 98% of the
time; it is then frozen and applied to the test split. The report shows the
acceptance rate, the accuracy of what was accepted, the number of wrong acceptances
and the review rate.

**Human effort saved.** A keystroke-equivalent cost model: entering a field by hand
costs 1.0; in a queued document, confirming a correct prefilled field costs 0.2 and
fixing a wrong or missing one costs 1.0; an accepted document costs nothing.
Effort saved = 1 minus assisted cost over manual cost. The same model backs
`/v1/stats` in the API, where the counts come from real reviews instead of the
oracle. Wrong acceptances are not charged by the model, which is why they are
reported next to it.

**Latency.** Wall time of `Pipeline.process` per page (preprocessing, OCR, layout,
extraction, validation), mean, median and 95th percentile, measured inside the
benchmark workers (4 processes on the build machine; single-page latency is the same
order on one process).

## Gate

`uv run python -m ledgerlens.bench --gate` compares the fresh results with
`benchmarks/thresholds.json` and exits non-zero on any violation. CI runs the gate on
a smaller split (160 test, 60 calibration pages) so the thresholds carry a margin
below the full-size numbers; the full-size numbers are the ones published.

## Reproducing

```
make setup
make bench            # full sizes, writes benchmarks/results.json and RESULTS.md
make bench-gate       # same, then checks thresholds
uv run python -m ledgerlens.bench --test-size 40 --cal-size 20 --jobs 2   # quick look
```

`--rebuild` regenerates the splits; otherwise the manifest fingerprint is checked and
cached pages are reused.
