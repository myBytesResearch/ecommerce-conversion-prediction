<!--
=============================================================================
 myBytes.com
 Copyright (c) 2026 myBytes GmbH. All rights reserved.
=============================================================================
-->

# ecommerce-conversion-prediction

**Companion repository to the myBytes Research note on honest e-commerce
session-conversion prediction.**

→ Methodology article (German): https://mybytes.com/research/conversion-vorhersage-ecommerce

---

## Scope

A common vendor pitch shows a session-conversion model with a near-perfect AUC.
We reproduce that on the **Data Mining Cup 2013** dataset (anonymised real data
from a **generic online shop** — the task PDF does not establish a vertical, so
this is general e-commerce, **not** a fashion study) and show where the
impressive number really comes from.

The honest finding: most of the eye-catching end-of-session AUC is **basket
tautology** — by the end of a session, whether the basket is full already nearly
tells you whether an order happened. Genuine *early* foresight is solid but far
more modest.

## What this repository reproduces

`scripts/01_conversion_early_vs_late.py` (session group-split, leak-free):

| Finding | Value |
|---|---|
| Sessions | 50,000 |
| Population order rate | **46.4 %** — the *curated competition population*, **not** a real funnel rate (real funnels are low single digits) |
| AUC, **early** (first transaction) | **0.848** |
| AUC, **full** (end-of-session state) | **0.961** |
| AUC, full **without** the basket block | **0.860** |
| Foresight-vs-tautology gap (full − early) | **0.114** |
| Basket-leakage contribution to AUC | **+0.101** |
| Top-decile lift (full model) | **×2.14** |

**The honest conclusion:** the ~0.96 "full session" AUC is largely an artefact of
the end-of-session basket state (remove the basket block and it falls to ~0.86).
Real, actionable early foresight — from the first interaction — is **~0.85**.
That is the number to plan an intervention on, not the tautological 0.96.

## What this repository does not contain

1. **No data.** DMC 2013 is third-party competition data; treat as
   non-redistributable. Fetch via `kagglehub`. See [`DATA.md`](DATA.md) /
   [`LICENSES.md`](LICENSES.md).
2. **No fabricated euro figures.** There is no clean in-data margin/AOV anchor, so
   we report predictability and the leakage gap (AUC, calibration, lift), not an
   invented cost-benefit.
3. **Not a fashion study.** The data is a generic webshop; the method transfers to
   any session-based shop, but this is not the Fashion-Intelligence pillar.

## Quickstart

```bash
git clone https://github.com/myBytesResearch/ecommerce-conversion-prediction.git
cd ecommerce-conversion-prediction
pip install -r requirements.txt
cp .env.example .env
python -c "import kagglehub; print(kagglehub.dataset_download('oscarm524/prediction-of-orders'))"
python scripts/01_conversion_early_vs_late.py
```

## Repository layout

```
notebooks/  Research notebook reproducing the headline numbers end-to-end
scripts/    The early-vs-late / basket-leakage analysis (session group-split)
results/    metrics.json (committed; the article's numbers)
figures/    01_early_vs_late.png (committed)
data/raw/   You place the fetched data here (gitignored)
DATA.md     Dataset identity, schema, scope caveats
LICENSES.md Code / data / library licensing
```

## Disclaimer

Methodological research on a public dataset, not business advice. The population
order rate is a competition artefact, not a transferable conversion benchmark.
