# Benchmark results

Generated 2026-10-25T04:30:41+00:00 with engine `atlas` on macOS-26.5.1-arm64-arm-64bit-Mach-O.

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

### Per field

| Field | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| closing_balance | 97.4% | 46.8% | 63.2% | 79 |
| currency | 99.3% | 71.8% | 83.3% | 400 |
| customer_name | 58.0% | 26.7% | 36.5% | 285 |
| document_date | 90.5% | 57.5% | 70.3% | 400 |
| document_number | 79.1% | 51.9% | 62.7% | 372 |
| due_date | 97.3% | 52.4% | 68.1% | 206 |
| gstin | 94.9% | 36.3% | 52.5% | 102 |
| iban | 100.0% | 34.3% | 51.1% | 105 |
| opening_balance | 97.6% | 50.6% | 66.7% | 79 |
| period_end | 100.0% | 57.0% | 72.6% | 79 |
| period_start | 100.0% | 57.0% | 72.6% | 79 |
| subtotal | 89.8% | 55.1% | 68.3% | 321 |
| tax | 89.8% | 55.1% | 68.3% | 321 |
| total | 86.9% | 53.9% | 66.5% | 321 |
| vendor_name | 47.4% | 45.5% | 46.4% | 400 |

### Per template

| Template | Docs | Field F1 | Line-item F1 | Transaction F1 |
|---|---|---|---|---|
| invoice_boxed | 35 | 57.5% | 39.4% | 0.0% |
| invoice_classic | 38 | 66.1% | 42.7% | 0.0% |
| invoice_letter | 45 | 64.5% | 43.5% | 0.0% |
| invoice_modern | 48 | 59.4% | 0.9% | 0.0% |
| invoice_tax | 40 | 62.1% | 33.5% | 0.0% |
| receipt_cafe | 33 | 64.8% | 37.9% | 0.0% |
| receipt_retail | 42 | 59.2% | 22.8% | 0.0% |
| receipt_thermal | 40 | 70.2% | 31.0% | 0.0% |
| statement_ledger | 36 | 66.3% | 0.0% | 27.3% |
| statement_ruled | 43 | 66.5% | 0.0% | 34.2% |

### Per severity

| Severity | Docs | Field F1 | OCR CER | Fully correct records |
|---|---|---|---|---|
| clean | 100 | 91.0% | 3.5% | 22.0% |
| heavy | 154 | 26.1% | 51.4% | 0.0% |
| light | 146 | 72.2% | 13.2% | 3.4% |

### Seen versus unseen fonts

| Fonts | Docs | Field F1 | OCR CER |
|---|---|---|---|
| unseen | 61 | 36.0% | 31.8% |
| seen | 339 | 68.0% | 23.9% |

### Acceptance at other thresholds (test split)

| Threshold | Accepted | Header exact | Fully exact | Accepted with a header error |
|---|---|---|---|---|
| 0.3 | 14.5% | 89.7% | 43.1% | 6 |
| 0.5 | 11.8% | 93.6% | 48.9% | 3 |
| 0.7 | 4.2% | 100.0% | 52.9% | 0 |
| 0.8 | 3.5% | 100.0% | 50.0% | 0 |
| 0.9 | 0.5% | 100.0% | 50.0% | 0 |
| 0.95 | 0.0% | 0.0% | 0.0% | 0 |
