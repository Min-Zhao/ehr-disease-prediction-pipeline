"""
generate_synthetic_data.py
──────────────────────────
Generate a synthetic EHR cohort for superimposed preeclampsia (PE) prediction.

The synthetic dataset mirrors the structure of a real chronic-hypertension
pregnancy cohort (ICD-10-CM codes O10–O11, I10) without containing any real
patient information.  All values are drawn from distributions calibrated to
published clinical literature.

Usage
─────
    python data/synthetic/generate_synthetic_data.py
    python data/synthetic/generate_synthetic_data.py --n 2000 --seed 99 --out data/synthetic/pe_cohort.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# ─── reproducibility ──────────────────────────────────────────────────────────
DEFAULT_SEED = 42
DEFAULT_N    = 1_500   # number of synthetic pregnancy episodes
DEFAULT_OUT  = Path(__file__).parent / "synthetic_pe_cohort.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def truncated_normal(rng, mean, std, low, high, size):
    a = (low  - mean) / std
    b = (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size,
                         random_state=rng)


def bernoulli(rng, p, size):
    return rng.binomial(1, p, size)


# ─────────────────────────────────────────────────────────────────────────────
# Cohort generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ─────────────────────────────────────────────────────────
    age = truncated_normal(rng, mean=29, std=6, low=12, high=50, size=n).round(1)

    race_choices  = ["Non-Hispanic White", "Non-Hispanic Black",
                     "Hispanic", "Asian", "Other/Unknown"]
    race_probs    = [0.38, 0.30, 0.20, 0.07, 0.05]
    race_ethnicity = rng.choice(race_choices, size=n, p=race_probs)

    insurance_choices = ["Medicaid", "Private", "Self-pay", "Other"]
    insurance_probs   = [0.55, 0.35, 0.05, 0.05]
    insurance_type    = rng.choice(insurance_choices, size=n, p=insurance_probs)

    parity    = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.30, 0.32, 0.22, 0.10, 0.06])
    gravidity = parity + rng.choice([0, 1, 2], size=n, p=[0.50, 0.35, 0.15])

    multiple_gestation     = bernoulli(rng, 0.03, n)
    gestational_age_wks    = truncated_normal(rng, mean=37.5, std=3, low=23, high=42, size=n).round(1)

    # ── Clinical history ─────────────────────────────────────────────────────
    # Chronic hypertension is the index condition for this cohort (all == 1)
    chronic_hypertension            = np.ones(n, dtype=int)
    chronic_hypertension_dur_yrs    = truncated_normal(rng, 3, 3, 0, 20, n).round(1)

    pregestational_diabetes         = bernoulli(rng, 0.12, n)
    gestational_diabetes            = bernoulli(rng, 0.10, n)
    renal_disease                   = bernoulli(rng, 0.08, n)
    autoimmune_disorder             = bernoulli(rng, 0.05, n)
    prior_pe                        = bernoulli(rng, 0.15, n)
    prior_preterm_birth             = bernoulli(rng, 0.12, n)
    smoking_status                  = bernoulli(rng, 0.10, n)
    bmi_prepregnancy                = truncated_normal(rng, 29, 7, 14, 70, n).round(1)

    # ── Vital signs ──────────────────────────────────────────────────────────
    sbp_t1 = truncated_normal(rng, 128, 14, 80, 200, n).round(0)
    dbp_t1 = truncated_normal(rng, 80,  10, 50, 130, n).round(0)
    sbp_t2 = truncated_normal(rng, 126, 14, 80, 200, n).round(0)
    dbp_t2 = truncated_normal(rng, 79,  10, 50, 130, n).round(0)
    sbp_max = np.maximum(sbp_t1, sbp_t2) + rng.integers(0, 20, size=n)
    dbp_max = np.maximum(dbp_t1, dbp_t2) + rng.integers(0, 15, size=n)

    # ── Labs ─────────────────────────────────────────────────────────────────
    protein_urine_24h  = truncated_normal(rng, 250, 450,  0, 6000, n).round(0)
    creatinine         = truncated_normal(rng, 0.85, 0.35, 0.3, 4.5, n).round(2)
    platelet_count     = truncated_normal(rng, 220, 65,  50, 450, n).round(0)
    alt                = truncated_normal(rng, 22, 18,  5, 300, n).round(0)
    ast                = truncated_normal(rng, 24, 18,  5, 300, n).round(0)
    uric_acid          = truncated_normal(rng, 4.8, 1.4, 1.0, 12.0, n).round(1)
    hemoglobin         = truncated_normal(rng, 11.5, 1.5, 7, 16, n).round(1)
    ldh                = truncated_normal(rng, 185, 60, 80, 800, n).round(0)

    # ── Outcome: superimposed PE ──────────────────────────────────────────────
    # Logistic model calibrated so ~25% baseline prevalence in chronic HTN cohort,
    # with clinically plausible risk factor contributions.
    log_odds = (
        -1.50                                             # intercept → ~25 % base rate
        + 0.04  * (age - 28)                             # older age ↑ risk
        + 0.35  * (race_ethnicity == "Non-Hispanic Black").astype(float)
        + 0.28  * prior_pe
        + 0.22  * renal_disease
        + 0.18  * pregestational_diabetes
        + 0.12  * autoimmune_disorder
        + 0.25  * (bmi_prepregnancy > 30).astype(float)
        + 0.015 * (sbp_max - 130).clip(0)
        + 0.020 * (protein_urine_24h / 100)
        - 0.015 * (platelet_count - 220).clip(None, 0) / 10
        + 0.010 * (ast - 25).clip(0) / 10
        + 0.010 * (uric_acid - 5).clip(0)
        + 0.005 * chronic_hypertension_dur_yrs
        + 0.20  * multiple_gestation
    )
    prob_pe        = 1 / (1 + np.exp(-log_odds))
    superimposed_pe = rng.binomial(1, prob_pe)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        # Demographics
        "age_at_delivery":                 age,
        "race_ethnicity":                  race_ethnicity,
        "insurance_type":                  insurance_type,
        "parity":                          parity,
        "gravidity":                       gravidity,
        "multiple_gestation":              multiple_gestation,
        "gestational_age_at_delivery":     gestational_age_wks,
        # Clinical history
        "chronic_hypertension":            chronic_hypertension,
        "chronic_hypertension_duration_yrs": chronic_hypertension_dur_yrs,
        "pregestational_diabetes":         pregestational_diabetes,
        "gestational_diabetes":            gestational_diabetes,
        "renal_disease":                   renal_disease,
        "autoimmune_disorder":             autoimmune_disorder,
        "prior_pe":                        prior_pe,
        "prior_preterm_birth":             prior_preterm_birth,
        "smoking_status":                  smoking_status,
        "bmi_prepregnancy":                bmi_prepregnancy,
        # Vital signs
        "sbp_first_trimester":             sbp_t1,
        "dbp_first_trimester":             dbp_t1,
        "sbp_second_trimester":            sbp_t2,
        "dbp_second_trimester":            dbp_t2,
        "sbp_max_antepartum":              sbp_max,
        "dbp_max_antepartum":              dbp_max,
        # Labs
        "protein_urine_24h_mg":            protein_urine_24h,
        "creatinine_mg_dl":                creatinine,
        "platelet_count_k_ul":             platelet_count,
        "alt_u_l":                         alt,
        "ast_u_l":                         ast,
        "uric_acid_mg_dl":                 uric_acid,
        "hemoglobin_g_dl":                 hemoglobin,
        "ldh_u_l":                         ldh,
        # Outcome
        "superimposed_pe":                 superimposed_pe,
    })

    # ── Introduce realistic missingness ──────────────────────────────────────
    missing_rates = {
        "protein_urine_24h_mg": 0.12,
        "ldh_u_l":              0.08,
        "creatinine_mg_dl":     0.05,
        "uric_acid_mg_dl":      0.07,
        "alt_u_l":              0.06,
        "ast_u_l":              0.06,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic EHR cohort for superimposed PE prediction."
    )
    parser.add_argument("--n",    type=int,  default=DEFAULT_N,   help="Number of episodes")
    parser.add_argument("--seed", type=int,  default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT,  help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating {args.n} synthetic pregnancy episodes (seed={args.seed}) …")
    df = generate_cohort(n=args.n, seed=args.seed)

    prevalence = df["superimposed_pe"].mean()
    print(f"  Superimposed PE prevalence : {prevalence:.1%}  ({df['superimposed_pe'].sum()} / {len(df)})")
    print(f"  Columns                    : {len(df.columns)}")
    print(f"  Missing values             : {df.isnull().sum().sum()} cells")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
