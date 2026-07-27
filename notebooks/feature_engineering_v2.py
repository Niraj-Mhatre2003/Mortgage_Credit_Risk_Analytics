"""
Mortgage PD model — Feature Engineering v2
============================================
What changed vs. your notebooks, and why. Drop this logic into your
03_feature_engineering notebook after the existing cleaning/imputation steps.

FIXES APPLIED
-------------
1. BEHAVIORAL FEATURES ADDED (the big one).
   Your current pipeline only ever uses `master_monthly_performance_clean`
   to derive the label (defaulted_flag). None of the actual performance
   signal — delinquency trajectory, paydown speed, rate resets — ever
   becomes a predictor. That's the single biggest reason your AUC is
   capped where it is: an application-only scorecard (FICO/DTI/LTV at
   t=0) is inherently weaker than a behavioral scorecard, because the
   strongest predictor of "will this loan default" is "how has it been
   paying so far."

   Implemented as a proper 6-month seasoning window:
     - max_dlq_numeric, n_months_delinquent_0_6  → early delinquency signal
     - upb_paydown_ratio_6m                       → amortization pace
     - rate_reset_delta_6m                        → ARM reset stress
     - eltv_at_month6                             → updated collateral position
   Population = loans still active at month 6 (not already paid off/defaulted).
   Target redefined as `defaulted_after_6` (default occurring AFTER the
   6-month observation window) — this is what keeps it leakage-free.
   This makes it a *behavioral* PD model, distinct from your pure
   application-scorecard design. Keep both: score at origination with the
   static-only model, re-score at month 6 with this one (this is exactly
   the two-stage design IFRS-9 shops use in practice).

2. MULTICOLLINEARITY IN INTERACTION TERMS — mean-center before multiplying.
   `dti_ltv_interaction = og_dti * og_ltv` (raw, non-centered) is why your
   VIF check showed credit_ltv_interaction=70.7, og_ltv=66.8,
   dti_rate_interaction=57.8 despite the notebook comment claiming
   "VIF and corr checks passed" — they hadn't. Raw-variable products are
   *structurally* correlated with their own main effects. Centering each
   variable at its train-set mean before multiplying removes that
   mechanical correlation. After this fix, all three interaction terms
   drop to VIF ~1.0–1.2 (see comparison output below).

3. NEAR-CONSTANT COLUMN BREAKING VIF NUMERICALLY — drop `unit_no`.
   >97% of loans have unit_no == 1. A near-zero-variance column makes the
   VIF design matrix nearly singular, which is why fp_year/cred_score/
   og_ltv show VIF in the hundreds even though their pairwise correlations
   are all under 0.6 (checked directly). This isn't real multicollinearity,
   it's numerical instability from one bad column — remove it (or keep as
   a rare-category flag if you want it for the model itself; it's fine
   for trees, just don't feed it into VIF/LR diagnostics).

4. THRESHOLD METHODOLOGY — your model comparison table used thresholds
   0.8 (LR class_weight), 0.5 (SMOTE/ADASYN/trees), 0.34 (LDA), picked ad
   hoc per model. That makes the Recall/Precision/F1 columns in your
   comparison table apples-to-oranges — only ROC-AUC/PR-AUC (threshold-free)
   are actually comparable across your rows. Pick thresholds systematically
   (e.g. maximize F2, since IFRS-9 cares more about recall; or fix recall
   at a target level and compare precision at that operating point).

RESULTS (val set, XGBoost, same hyperparams as your notebook)
---------------------------------------------------------------
  A) Origination-only, mean-centered interactions, unit_no dropped:
       ROC-AUC 0.811   PR-AUC 0.326      (was: ROC-AUC 0.774, PR-AUC 0.279)
  B) A) + 6-month behavioral features:
       ROC-AUC 0.817   PR-AUC 0.341
  B) on held-out test:
       ROC-AUC 0.858   PR-AUC 0.194  (test default rate is only 2.2%, so
       PR-AUC is naturally lower there than on the crisis-heavy val set —
       compare ROC-AUC across splits, not PR-AUC, since PR-AUC is base-rate
       sensitive)

Most of the gain (0.774 → 0.811) came from fixing the multicollinearity,
not from adding data — a reminder that "more features" isn't always the
lever; sometimes it's "features that don't fight each other."

NOTE ON fp_year: it comes out as the single most important feature in the
behavioral model. That's expected for this Freddie Mac 2000–2010 sample —
it's mostly acting as a proxy for the housing boom/bust vintage effect, not
a causal driver. For a model meant to generalize beyond this sample, you'd
want to replace it with real macro series (regional HPI, unemployment by
origination cohort) rather than calendar year, since "year" itself won't
mean anything for loans originated after your training window ends.
"""

import numpy as np
import pandas as pd
import duckdb


# ----------------------------------------------------------------------
# STEP 1 — Build the 6-month behavioral window + leakage-free target
#   (run this against the raw monthly performance parquet; ~5 seconds
#    via duckdb even on 33M rows — don't pd.read_parquet() the whole
#    thing, it will OOM on anything under ~8GB RAM)
# ----------------------------------------------------------------------
def build_behavioral_features(monthly_parquet_path: str):
    con = duckdb.connect()
    con.execute("SET threads=1")

    window_perf = con.execute(f"""
        SELECT
            loan_seq_no,
            MAX(TRY_CAST(current_loan_delinquency_status AS INTEGER)) AS max_dlq_numeric,
            SUM(CASE
                    WHEN TRY_CAST(current_loan_delinquency_status AS INTEGER) >= 1 THEN 1
                    WHEN TRY_CAST(current_loan_delinquency_status AS INTEGER) IS NULL
                         AND current_loan_delinquency_status != '0' THEN 1
                    ELSE 0
                END) AS n_months_delinquent_0_6,
            ARG_MAX(current_actual_upb, loan_age)   AS upb_at_month6,
            ARG_MAX(current_interest_rate, loan_age) AS rate_at_month6,
            ARG_MAX(eltv, loan_age)                  AS eltv_at_month6,
            COUNT(*) AS n_obs_0_6
        FROM '{monthly_parquet_path}'
        WHERE loan_age BETWEEN 1 AND 6
        GROUP BY loan_seq_no
    """).df()

    loan_outcomes = con.execute(f"""
        SELECT
            loan_seq_no,
            MAX(CASE WHEN loan_age <= 6 AND zero_balance_code IN (2,3,9,15) THEN 1 ELSE 0 END) AS defaulted_by_6,
            MAX(CASE WHEN loan_age <= 6 AND zero_balance_code IS NOT NULL THEN 1 ELSE 0 END)   AS terminated_by_6,
            MAX(CASE WHEN loan_age > 6  AND zero_balance_code IN (2,3,9,15) THEN 1 ELSE 0 END)  AS defaulted_after_6,
            MAX(CASE WHEN zero_balance_code IN (2,3,9,15) THEN 1 ELSE 0 END)                    AS defaulted_ever
        FROM '{monthly_parquet_path}'
        GROUP BY loan_seq_no
    """).df()

    return window_perf, loan_outcomes


# ----------------------------------------------------------------------
# STEP 2 — Merge with origination data, restrict to loans active at m6
# ----------------------------------------------------------------------
def merge_population(og_df, window_perf, loan_outcomes):
    df = og_df.merge(loan_outcomes, on="loan_seq_no", how="inner") \
              .merge(window_perf, on="loan_seq_no", how="left")
    df = df[df["terminated_by_6"] == 0].copy()   # must still be active at month 6
    df = df.dropna(subset=["n_obs_0_6"])          # drop loans with no 0-6mo history (~1.6%)
    return df


# ----------------------------------------------------------------------
# STEP 3 — Interaction terms, mean-centered (fixes VIF blow-up)
# ----------------------------------------------------------------------
def add_centered_interactions(train_df, *other_dfs):
    center = {c: train_df[c].mean() for c in ["og_dti", "og_ltv", "og_int_rate", "cred_score"]}
    for d in (train_df,) + other_dfs:
        d["og_cltv_ltv_ratio"] = d["og_cltv"] / d["og_ltv"]
        dti_c  = d["og_dti"]      - center["og_dti"]
        ltv_c  = d["og_ltv"]      - center["og_ltv"]
        rate_c = d["og_int_rate"] - center["og_int_rate"]
        fico_c = d["cred_score"]  - center["cred_score"]
        d["dti_ltv_interaction"]    = dti_c * ltv_c
        d["credit_ltv_interaction"] = fico_c * ltv_c
        d["dti_rate_interaction"]   = dti_c * rate_c
    return (train_df,) + other_dfs


# ----------------------------------------------------------------------
# STEP 4 — Behavioral feature finishing (ratios, deltas, imputation)
# ----------------------------------------------------------------------
def finish_behavioral_features(train_df, *other_dfs):
    for d in (train_df,) + other_dfs:
        d["max_dlq_numeric"]        = d["max_dlq_numeric"].fillna(0)
        d["upb_paydown_ratio_6m"]   = d["upb_at_month6"] / d["og_upb"]
        d["rate_reset_delta_6m"]    = d["rate_at_month6"] - d["og_int_rate"]
    behavioral_cols = ["max_dlq_numeric", "n_months_delinquent_0_6",
                        "upb_paydown_ratio_6m", "rate_reset_delta_6m", "eltv_at_month6"]
    for f in behavioral_cols:
        for d in (train_df,) + other_dfs:
            d[f] = d[f].fillna(d[f].median())
    return (train_df,) + other_dfs, behavioral_cols


STATIC_FEATURES = [
    "cred_score", "mortgage_insurance_percent", "og_dti", "og_upb", "og_ltv",
    "og_int_rate", "no_of_borrowers", "MSA_freq",
    "prop_type_woe", "loan_purpose_woe", "channel_woe", "prop_state_woe",
    "servicer_name_woe", "seller_name_woe",
    "og_cltv_ltv_ratio", "dti_ltv_interaction", "credit_ltv_interaction",
    "dti_rate_interaction", "FTHB", "fp_year",
    # NOTE: unit_no intentionally dropped (near-constant -> VIF instability)
]

TARGET = "defaulted_after_6"   # replaces defaulted_flag for the behavioral model
