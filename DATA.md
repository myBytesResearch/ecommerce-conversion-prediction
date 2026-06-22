# Data - E-commerce Conversion (verified)

Raw data is NOT committed (`data/raw/` gitignored): competition / Kaggle terms
forbid redistribution. Reproduced from the kagglehub cache. See `LICENSES.md`.

## Dataset (verified 2026-06-22)
**Data Mining Cup 2013 - prediction of orders.** Kaggle slug
`oscarm524/prediction-of-orders`. Per the task PDF: a **generic online shop**
(anonymised real shop data) - NOT fashion-specific. This repo therefore frames
the study as **general e-commerce conversion**, not Fashion-Intelligence.

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('oscarm524/prediction-of-orders'))"
```

## Schema (verified)
- `transact_train.txt` (44 MB, **`|`-separated**, missing = `?`). 429,013
  transaction (click) rows across **50,000 sessions**, 24 columns.
- Target: **`order` (y/n)**, a per-session label (constant within a session;
  asserted in the script).
- Per-session **order rate = 46.4%** - this is the **curated competition
  population**, NOT a real funnel conversion rate (real funnels are low single
  digits). Always labelled as such.
- Basket block (`bCount, bMinPrice, bMaxPrice, bSumPrice, bStep`) encodes the
  end-of-session state and is the leakage-prone group the study isolates.

## Method note
Validation splits **by `sessionNo`** (group split), never by transaction row.
"Early" = each session's first transaction; "full" = last transaction (end
state). The early-vs-full AUC gap separates real foresight from end-state
tautology. License: see `LICENSES.md`.
