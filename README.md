# Mortgage Credit Risk Analytics

A credit risk modelling project built on Freddie Mac's public Single Family Loan Level Dataset. The project estimates the probability of default (PD), loss given default (LGD), and exposure at default (EAD) for a portfolio of mortgages, combines these into an Expected Loss estimate, and examines how that estimate changes under a stressed macroeconomic scenario.

## Scope

This project applies the Expected Loss decomposition, EL equals PD times LGD times EAD, that underlies credit risk measurement generally, including regulatory frameworks such as IFRS 9 and CECL. The stress testing here is a single scenario analysis, calibrated from the gap between the portfolio's own benign and crisis vintages, used to show how Expected Loss responds to a downturn. It is a scenario analysis rather than a full IFRS 9 or CECL provisioning calculation, which would additionally require staged exposures and probability weighted forward looking scenarios.

## Overview

The dataset covers 545,101 mortgages originated between 2000 and 2010, with 33 million plus associated monthly performance records. The notebooks are organised as a sequence of separate modelling decisions rather than one script, so that the reasoning behind each choice, such as the feature set, the treatment of class imbalance, and the validation scheme, is visible and can be checked.

## Data

- Source: Freddie Mac Single Family Loan Level Dataset (public, sampled). See https://www.freddiemac.com/research/datasets. The raw origination and monthly performance files are not included in this repository, as Freddie Mac's terms require users to download the data directly. See Setup below.
- Scope: origination years 2000 to 2010, 545,101 loans, 33 million plus monthly performance records.
- Auxiliary macroeconomic data, tracked in `Data/raw/`:
  - `hpi_at_state.xlsx`, Freddie Mac's state level House Price Index, used to re-index loan to value ratio at the observation date.
  - `fred_unemployment_<state range>.csv`, five files, state level unemployment rate (FRED series code `{state}URN`) obtained from FRED. The series is split across five downloads because FRED limits the number of series per export.
- Population scope: of the 545,101 loans, 471,654 (86.5 per cent) survive to a 12 month observation window and form the modelling population for the behavioural PD, LGD, and Expected Loss stages. Loans that default (1.0 per cent) or prepay (12.5 per cent) within the first year are excluded from this population, as they do not generate the 12 month performance history the behavioural features require.

## Methodology

1. Preprocessing (`01_pre_processing.ipynb`). Sentinel value cleaning, since Freddie Mac uses numeric placeholders such as 999 or 9999 for missing values, dtype normalisation across vintages, and default labelling using a Basel style definition (90 or more days past due, or a terminal credit event zero balance code).
2. Exploratory analysis (`02_EDA.ipynb`). Default rate, delinquency, and macroeconomic (HPI) patterns across origination vintages 2000 to 2010.
3. Feature engineering (`03_feature_engineering.ipynb`). Weight of evidence and information value encoding for the linear model, a separate raw feature set for tree based models, and leakage safe fitting, meaning transforms are fitted on training years only and applied without refitting to validation, test, and out of time vintages.
4. Baseline PD model, static features (`04_baseline_pd_static.ipynb`). Logistic regression, random forest, and XGBoost benchmarked on point in time underwriting features (credit score, LTV, DTI, interest rate). Retained in the repository to document the model selection reasoning rather than as the final model. Walk forward validation shows that discrimination is not stable across the credit cycle, which motivates step 5.
5. Behavioural PD model (`05_behavioural_modelling.ipynb`). Replaces static features with each loan's own 12 month performance history (delinquency pattern, re-indexed LTV using updated HPI, and macroeconomic variables at the observation date). XGBoost is selected as the final model, though a gradient boosting classifier performs marginally better on the same leaderboard (see Key Results). Stability across vintages is checked using the Population Stability Index.
6. LGD model (`06_lgd_modelling.ipynb`). A two stage model, a classifier for P(loss greater than zero) followed by a severity regression, fitted on 16,435 defaulted loans. Loss severity is driven primarily by indexed LTV at default.
7. Expected Loss and stress testing (`07_expected_loss_modelling.ipynb`). Expected Loss is calculated per loan as PD times LGD times EAD and aggregated to the portfolio level, then recalculated under a single stressed scenario, with PD and LGD shocks calibrated from the gap between the portfolio's own benign and crisis vintages.

## Key Results

All figures below are taken directly from the printed output of the corresponding notebook cell, so they can be reproduced by re-running the notebooks in order.

| Component | Approach | Result |
|---|---|---|
| PD, baseline, static features | Logistic regression, random forest, XGBoost | AUC 0.77 to 0.79 on the 2007 test vintage. Discrimination degrades across vintages under walk forward validation, which is why this model is not used further. |
| PD, final, behavioural features | XGBoost, tuned via Optuna | AUC 0.8528, KS 0.5222, Brier 0.1206, PR AUC 0.7372 (2007 test, trained on 2004 to 2005, threshold frozen on the 2006 validation set). After isotonic calibration on the validation set: AUC 0.8524, Brier 0.1203. On the same leaderboard, a gradient boosting classifier scored marginally higher (AUC 0.8533, PR AUC 0.7376). AUC across all 2000 to 2010 vintages ranges from 0.819 to 0.879. |
| LGD | Two stage model, classifier times severity regression | Combined out of time test R squared of 0.446, on all defaulted loans (the notebook itself reports this combined figure as the correct one, rather than the severity stage taken alone). |
| Expected Loss | PD times LGD times EAD | Total Expected Loss of approximately 3.04 billion dollars on approximately 78.1 billion dollars of exposure, or 3.90 per cent of exposure, under the base case. |
| Stress scenario | A single deterministic shock based on the 2006 to 2008 crisis vintages | Mean PD rises by a factor of 2.06, from 0.116 to 0.215. Expected Loss as a share of exposure rises from 3.90 per cent to 8.42 per cent. |

## Interpretation of the Expected Loss results

A few observations from the decomposition in notebook 7 are worth noting.

- Loss is concentrated in a small share of the book. Splitting the portfolio into PD deciles of equal loan count, the highest PD decile alone accounts for 47.6 per cent of total portfolio Expected Loss, and the top two deciles together account for around 65 per cent.
- Loss is also concentrated by LTV. Loans with an indexed LTV between 70 and 80 make up the largest LTV band by loan count and account for 56.4 per cent of total Expected Loss, more than any other band.
- Classifying vintages by their early life delinquency behaviour, rather than by their eventual lifetime default rate alone, changes the picture for the 2004 vintage. Based on lifetime default rate, 2004 looks similar to the crisis vintages. Its delinquency rate at 36 months on book, however, is lower than every vintage from 2000 to 2002 and close to the benign group overall. This suggests that 2004 loans were underwritten in broadly the same way as earlier, benign vintages, and their elevated lifetime default rate mainly reflects the fact that they were 4 to 5 years into their life when the 2008 downturn arrived, rather than weaker underwriting at origination. On this basis, vintages 2000 to 2004 are grouped as benign (mean Expected Loss of 2.99 per cent of exposure), 2005 as transitional (4.26 per cent), 2006 to 2008 as crisis (6.35 per cent), and 2009 to 2010 as post crisis (2.12 per cent).

## Why behavioural features were used instead of only static underwriting features

A model trained only on point in time underwriting data, such as credit score, LTV, and DTI at origination, cannot observe how a borrower's risk changes after the loan is issued, since those features never update. In this dataset, discrimination based on static features decays specifically in the vintages that matter most for a risk framework, namely the years leading into and through the 2006 to 2008 crisis. Replacing static features with a rolling 12 month performance window, covering delinquency trend, updated LTV, and macroeconomic conditions at the observation date, allows the final model to retain discrimination on vintages it was not trained on.

## Repository structure

```
Mortgage_Credit_Risk_Analytics/
├── Data/
│   └── raw/
│       ├── hpi_at_state.xlsx
│       └── fred_unemployment_*.csv        (five files, state range in filename)
│       (Data/processed/ and Data/features/ are not tracked; regenerate using
│       notebooks 01 and 03, see Setup)
├── notebooks/
│   ├── 01_pre_processing.ipynb            sentinel cleaning, dtype fixes, default labelling
│   ├── 02_EDA.ipynb                       exploratory analysis, vintage patterns
│   ├── 03_feature_engineering.ipynb       WOE/IV encoding, raw feature set, leakage safe fitting
│   ├── 04_baseline_pd_static.ipynb        static feature PD baseline and walk forward validation
│   ├── 05_behavioural_modelling.ipynb     final PD model, 12 month behavioural features, XGBoost
│   ├── 06_lgd_modelling.ipynb             two stage LGD model on defaulted loans
│   ├── 07_expected_loss_modelling.ipynb   EL = PD x LGD x EAD, vintage/PSI analysis, stress test
│   └── final_behavioural_xgb.pkl          saved final PD model from notebook 05
├── user_guide.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Raw Freddie Mac data is not included in this repository, since their terms of use require direct download. Obtain the sample dataset from the Freddie Mac Single Family Loan Level Dataset portal (https://www.freddiemac.com/research/datasets), place the raw origination and performance files under `Data/raw/`, and run the notebooks in order. Each notebook reads the previous notebook's saved output, stored as parquet under `Data/processed/` and `Data/features/`, rather than re-reading the raw text files, so the raw files only need to be parsed once.

## Notes on methodology choices

- Default definition. A loan is labelled as default if it reaches 90 or more days past due at any point, or exits via a terminal credit event zero balance code (third party sale, short sale or charge off, or REO disposition). This definition is applied consistently in every notebook that constructs a label.
- Leakage discipline. Every fitted object, including WOE bins, imputation medians, scalers, and IV based feature selection, is fitted on training years only and applied, without refitting, to validation, test, and out of time vintages.
- Population scope. The behavioural PD, LGD, and Expected Loss stages cover the 471,654 of 545,101 loans that survive to a 12 month behavioural observation window. Loans that default or prepay within the first year are excluded from this population rather than scored. See `05_behavioural_modelling.ipynb` for the exact breakdown.

## Authors

Niraj Mhatre, github.com/Niraj-Mhatre2003
Manya Gupta, github.com/manyagupta-21
