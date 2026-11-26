# Plastic Waste Generation Across Indian States
### A Cluster and Regression Assessment of Reported Data

This repository holds the data, code, results and figures for the report in [`paper/Plastic_Waste_Report_Final.pdf`](paper/Plastic_Waste_Report_Final.pdf).
Everything in the paper is produced by one script, `analysis.py`.

## What the study does
Official CPCB figures for plastic waste generation (35 states/UTs, 2016-17 to 2020-21) are incomplete and highly skewed. The study:

1. **Clusters** jurisdictions with K-means on standardised mean generation and endpoint growth (K = 3 by silhouette).
2. **Tests** that clustering with seed and bootstrap stability, null models (random and permuted features) and alternative preprocessing.
3. **Regresses** log mean generation on log population and urban share (mean-only baseline, OLS, ridge, lasso, random forest; leave-one-out and repeated 5-fold CV).
4. **Flags** jurisdictions whose reported totals are far from what population suggests (e.g. Tripura, Sikkim).

The results are descriptive and associational. They do not establish causes or policy effects.

## Key results (all reproduced by `analysis.py`)
| Quantity | Value |
|---|---|
| Silhouette at K = 3 | 0.790 (K = 2: 0.698, K = 4: 0.628, K = 5: 0.617, K = 6: 0.592) |
| Cluster sizes (0 / 1 / 2) | 24 / 3 / 8 |
| Seed stability (100 seeds) | ARI = 1.00 |
| Bootstrap stability (500 resamples) | mean ARI = 0.994 |
| Silhouette, permuted-feature null | mean 0.785 (95th percentile 0.808), so the observed 0.790 is within the null |
| OLS fit | R² = 0.811 (adjusted 0.800) |

**Cross-validated regression (target: log10 mean generation; lower RMSE is better)**

| Model | LOOCV RMSE | 5-fold x 20 RMSE (mean ± sd) | LOOCV R² |
|---|---|---|---|
| Mean-only baseline | 1.072 | 1.075 ± 0.020 | -0.06 |
| OLS | 0.492 | 0.499 ± 0.020 | 0.78 |
| Ridge | **0.489** | 0.496 ± 0.019 | 0.78 |
| Lasso | 0.509 | 0.515 ± 0.023 | 0.76 |
| Random forest | 0.556 | 0.575 ± 0.032 | 0.71 |

**Largest ridge LOOCV residuals** (observed minus predicted log10): Tripura -2.13, Sikkim -0.96, Goa +0.69, Telangana +0.53, Delhi NCR -0.52, Jammu and Kashmir +0.48.
Tripura and Sikkim report about 0.01 and 0.11 kg per person per year, against 20.1 kg for Goa. That range is more consistent with differences in reporting coverage than in true generation, although this study cannot confirm it.

The K-means typology is reproducible but **not** robust: log or rank transformation of the features changes the partition almost completely (ARI 0.05-0.16), and its three-member group's mean growth of 47,748% is an artefact of very low baselines.
## Repository contents
| Path | Content |
|---|---|
| `analysis.py` | Single script that reproduces every number, table and figure in the paper |
| `data/plastic_waste_cpcb.csv` | CPCB state/UT plastic waste generation, tonnes per annum, 2016-17 to 2020-21 (empty cell = NA) |
| `data/state_covariates_census2011.csv` | Population and urban share (Census of India 2011) used in the regression |
| `results/` | Generated tables (`table*.csv`), state-level results and `summary.json` |
| `figures/` | Generated figures 1-6 |
| `paper/Plastic_Waste_Report_Final.pdf` | The final report |
| `requirements.txt` | Python dependencies (the versions used for the reported results) |

## Run it yourself
```bash
git clone https://github.com/Adityaraj84/ML-practical.git
cd ML-practical
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```
Paths are project-relative and `random_state = 42` is used throughout. Runtime is about one minute. The script overwrites `results/` and `figures/`.

## Data sources
* **Plastic waste:** Rajya Sabha Unstarred Question No. 1159 (answered 5 December 2024), Annexure I, on the OGD Platform India:
  <https://www.data.gov.in/resource/stateut-wise-details-data-plastic-waste-generation-central-pollution-control-board-cpcb>
  (identical table in PIB Release ID 1943210, 27 July 2023). Values are in tonnes per annum; the figures are *reported* by State Pollution Control Boards and consolidated by the CPCB, not independently measured. Resources on the OGD Platform are normally released under the Government Open Data License - India (GODL-India); see the resource page for the terms that apply. Accessed and cross-checked against the PIB release on 26 November 2026.
* **Covariates:** Census of India 2011 (Office of the Registrar General & Census Commissioner, India). Andhra Pradesh uses the post-2014 residual state, "Delhi NCR" uses the NCT of Delhi, and the two Daman/Dadra units are merged.

## Limitations
* The source data are incomplete (27 of 35 jurisdictions miss at least one year), and growth windows differ in length.
* Mean generation averages different years for different jurisdictions; the regression was not repeated on a common window.
* The Census 2011 covariates precede the 2016-21 window, and administrative boundaries have changed.
* Only two demographic predictors are used; income, industry, tourism and collection coverage are not included.
* One reporting window and 35 units cannot support causal or timing claims.

## Use of AI tools
An AI assistant (Claude, Anthropic) was used during revision to help locate and cross-check the public data table, to write and run analysis code, to draft and edit text, and to format the report. The author is responsible for the content and for verifying the data, references and statements. See the "Use of AI Tools" section of the paper.

