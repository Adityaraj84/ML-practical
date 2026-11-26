# Plastic Waste Generation Across Indian States: A Cluster and Regression Assessment of Reported Data

Course project (Machine Learning, B.Sc. (H) Computer Science, Ramanujan College, University of Delhi).
Author: Adityaraj Maurya. Supervisor: Dr. Nikhil Kumar Rajput.

## What this repository contains
| Path | Content |
|---|---|
| `analysis.py` | Single script that reproduces every number, table and figure in the paper |
| `data/plastic_waste_cpcb.csv` | CPCB state/UT plastic waste generation, tonnes per annum, 2016-17 to 2020-21 (empty cell = NA) |
| `data/state_covariates_census2011.csv` | Population and urban share (Census of India 2011) used in the regression |
| `results/` | Generated tables (`table*.csv`), state-level results and `summary.json` |
| `figures/` | Generated figures 1-6 |
| `paper/Plastic_Waste_Report_Final.pdf` | The final report these files accompany |
| `requirements.txt` | Pinned Python dependencies (the versions used for the reported results) |

## Run from a fresh clone
```bash
git clone <your-repository-url>
cd <repository-folder>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```
All paths are project-relative and `random_state = 42` is used throughout. Runtime is about one to two minutes.

## Data sources and licence
* Plastic waste: Rajya Sabha Unstarred Question No. 1159 (answered 5 December 2024), Annexure I, as listed on the OGD Platform India:
  https://www.data.gov.in/resource/stateut-wise-details-data-plastic-waste-generation-central-pollution-control-board-cpcb
  (identical table in PIB Release ID 1943210, 27 July 2023). Licence: OGD Platform resources are normally released under the Government Open Data License - India (GODL-India); see the resource page for the terms that apply.
* Covariates: Census of India 2011 (Office of the Registrar General & Census Commissioner, India).
  **Verify every value in `state_covariates_census2011.csv` against the Primary Census Abstract before relying on it.**
  Notes: Andhra Pradesh uses the post-2014 residual state, Delhi NCR uses NCT of Delhi, and the two Daman/Dadra units are merged.

## Notes
* Data accessed and cross-checked against the PIB release on 26 November 2026.
* Use of AI tools: see the "Use of AI Tools" section of the paper.
