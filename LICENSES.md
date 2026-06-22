# Licenses - Code, Data, Libraries

Code + result artifacts only. The raw dataset is NOT included; fetch it yourself.

## 1 - Repository code
MIT (see `LICENSE`). Use, modify, commercialise freely; no warranty.

## 2 - Data source (read carefully)
**Data Mining Cup 2013 - "prediction of orders"** (prudsys). Public Kaggle mirror
slug `oscarm524/prediction-of-orders`. The task PDF describes a **generic online
shop** (not a specific vertical). Treat it as **third-party competition data,
non-redistributable**: original Data Mining Cup terms + Kaggle dataset terms
apply. Check the Kaggle page for the uploader's declared license before any use
beyond local research. We ship **code only** (`data/raw/` gitignored); you fetch
it via kagglehub under your own terms.

## 3 - Result artifacts
`results/*.json` and `figures/*.png` are our derived results, committed so the
article's numbers are inspectable. Republishing derived results is fine;
republishing the raw data is not.

## 4 - Python libraries
numpy, pandas, scikit-learn (BSD-3) - matplotlib (PSF/BSD) - kagglehub (Apache-2.0,
data terms apply) - python-dotenv (BSD-3). All permissive, commercial use allowed.

If unclear for your use case, consult your own legal counsel. Documentation, not
legal advice.
