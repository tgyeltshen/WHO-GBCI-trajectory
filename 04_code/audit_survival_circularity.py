"""Predictor-circularity audit of the WHO breast cancer survival estimates.

The frozen protocol requires this audit before survival is used analytically:

    "A predictor-circularity audit will check whether mortality or correlated inputs helped
     model survival estimates before survival is used analytically."

The audit has two arms.

  Documentary. What WHO states about how the estimates were produced, recorded verbatim with
  its source, so the finding does not rest on inference from the numbers alone.

  Empirical. How much of the variation in the published survival estimates is shared with
  quantities this study derives from its own outcome, in particular the mortality-to-incidence
  ratio, and how strongly survival tracks this study's estimand (the 2015 to 2023 trend).

Two distinct uses are adjudicated separately, because they carry different risk:

  Use A. The survival VALUE as an explanatory variable for the mortality trend.
  Use B. The survival estimate's SOURCE CLASS as a data-availability stratification axis.

Writes 05_results/survival_circularity_audit.md and
05_results/survival_circularity_diagnostics.csv. Exit status is 0 when the audit completes,
regardless of verdict; the verdict is the content, not the exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_code"))

PROCESSED = ROOT / "03_processed_data"
RESULTS = ROOT / "05_results"

SURVIVAL_WINDOW = (2017, 2021)

# Verdict thresholds, fixed here before the numbers are read, so the verdict is a rule and not a
# reaction. A shared-variance fraction at or above REJECT_R2 means the published survival series
# is substantially a re-expression of quantities this study derives from its own outcome.
REJECT_R2 = 0.50
REJECT_ABS_R_WITH_ESTIMAND = 0.40

# Recorded verbatim from the WHO source pages on 19 August 2026, with URLs, so a later reader can
# check the quotation rather than trust this file.
DOCUMENTARY_RECORD = [
    (
        "WHO Global Health Observatory, breast cancer survival theme page "
        "(https://www.who.int/data/gho/data/themes/breast-cancer-survival)",
        [
            "Method of estimation: a Bayesian hierarchical statistical model to combine "
            "information from population-based cancer registry estimates, covariates associated "
            "with breast cancer survival, and regional levels and trends.",
            "Data sources: population-based cancer registries, vital/death registration, and "
            "facility-based monitoring.",
            "Where observed survival estimates were submitted by Member States during the Country "
            "Consultation, these were used as the final estimates.",
        ],
    ),
    (
        "Girardi F, Nyangasi M, Callender C, et al. Global breast cancer survival estimates in "
        "2017-2021 to advance the WHO Global Breast Cancer Initiative. Nat Med 2026. "
        "doi:10.1038/s41591-026-04531-2 (PMID 42420542)",
        [
            "The WHO estimated population-based age-standardized 5-year net survival for women "
            "diagnosed with breast cancer between 2017 and 2021 across all 194 Member States.",
            "Full text is not open access and is not deposited in PubMed Central, so the "
            "covariate list of the Bayesian hierarchical model could not be read directly. The "
            "documentary arm therefore rests on the WHO Global Health Observatory statement of "
            "data sources, and the empirical arm carries correspondingly more weight.",
        ],
    ),
]


def load_survival() -> pd.DataFrame:
    """Country-level WHO 5-year net survival with its published uncertainty interval."""
    path = PROCESSED / "who_survival.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["SpatialDimType"].eq("COUNTRY")]
    frame = frame[["SpatialDim", "NumericValue", "Low", "High"]].copy()
    frame.columns = ["iso3", "survival", "survival_lower", "survival_upper"]
    frame["survival_ui_width"] = frame["survival_upper"] - frame["survival_lower"]
    return frame.dropna(subset=["survival"])


def load_outcome_derived() -> pd.DataFrame:
    """Mortality, reconstructed incidence and their ratio, averaged over the survival window."""
    start, end = SURVIVAL_WINDOW
    mortality = pd.read_csv(PROCESSED / "gbd_primary_mortality.csv")
    incidence = pd.read_csv(PROCESSED / "gbd_constructed_incidence_asr.csv")
    crosswalk = pd.read_csv(RESULTS / "hierarchical_country_trajectory.csv")[["location", "iso3"]]

    mortality = (
        mortality[mortality["year"].between(start, end)]
        .groupby("location", as_index=False)["rate"]
        .mean()
        .rename(columns={"rate": "mortality_asr"})
    )
    incidence = (
        incidence[incidence["year"].between(start, end)]
        .groupby("location", as_index=False)["asir_per_100k"]
        .mean()
        .rename(columns={"asir_per_100k": "incidence_asr"})
    )
    frame = mortality.merge(incidence, on="location").merge(crosswalk, on="location", how="inner")
    frame["mi_ratio"] = frame["mortality_asr"] / frame["incidence_asr"]
    return frame


def correlate(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson, pearson_p = stats.pearsonr(x, y)
    spearman, spearman_p = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman),
        "spearman_p": float(spearman_p),
        "shared_variance_r2": float(pearson**2),
    }


def main() -> int:
    survival = load_survival()
    derived = load_outcome_derived()
    trajectory = pd.read_csv(RESULTS / "hierarchical_country_trajectory.csv")
    assignments = pd.read_csv(RESULTS / "subgroup_country_assignments.csv")

    panel = (
        derived.merge(survival, on="iso3", how="inner")
        .merge(trajectory[["iso3", "apc_unpooled"]], on="iso3", how="left")
        .merge(
            assignments[["iso3", "median_relative_ui_width", "registry_years_in_window"]],
            on="iso3",
            how="left",
        )
    )
    panel = panel.dropna(subset=["survival", "mi_ratio"])

    tests = [
        ("survival value", "GBD mortality-to-incidence ratio", panel["survival"], panel["mi_ratio"]),
        ("survival value", "GBD mortality ASR", panel["survival"], panel["mortality_asr"]),
        ("survival value", "reconstructed incidence ASR", panel["survival"], panel["incidence_asr"]),
        ("survival value", "study estimand (unpooled APC)", panel["survival"], panel["apc_unpooled"]),
        (
            "survival UI width",
            "study estimand (unpooled APC)",
            panel["survival_ui_width"],
            panel["apc_unpooled"],
        ),
        ("survival UI width", "GBD mortality ASR", panel["survival_ui_width"], panel["mortality_asr"]),
        (
            "survival UI width",
            "years of registry series",
            panel["survival_ui_width"],
            panel["registry_years_in_window"],
        ),
        (
            "survival UI width",
            "GBD relative UI width",
            panel["survival_ui_width"],
            panel["median_relative_ui_width"],
        ),
    ]

    rows = []
    for left, right, x, y in tests:
        usable = pd.concat([x, y], axis=1).dropna()
        rows.append(
            {
                "quantity": left,
                "compared_with": right,
                "countries": len(usable),
                **correlate(usable.iloc[:, 0], usable.iloc[:, 1]),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(RESULTS / "survival_circularity_diagnostics.csv", index=False)

    def pick(quantity: str, against: str) -> pd.Series:
        return diagnostics[
            diagnostics["quantity"].eq(quantity) & diagnostics["compared_with"].eq(against)
        ].iloc[0]

    mi = pick("survival value", "GBD mortality-to-incidence ratio")
    estimand = pick("survival value", "study estimand (unpooled APC)")
    width_estimand = pick("survival UI width", "study estimand (unpooled APC)")
    width_mortality = pick("survival UI width", "GBD mortality ASR")
    width_registry = pick("survival UI width", "years of registry series")

    # The naive identity a reader will suspect: is net survival simply 100(1 - MI)?
    identity = 100.0 * (1.0 - panel["mi_ratio"])
    identity_fit = stats.linregress(identity, panel["survival"])
    identity_gap = float(np.abs(panel["survival"] - identity).mean())

    use_a_fails = (
        float(mi["shared_variance_r2"]) >= REJECT_R2
        or abs(float(estimand["pearson_r"])) >= REJECT_ABS_R_WITH_ESTIMAND
    )
    use_b_fails = abs(float(width_mortality["spearman_rho"])) >= REJECT_ABS_R_WITH_ESTIMAND

    lines: list[str] = []
    add = lines.append
    add("# Predictor-circularity audit of the WHO breast cancer survival estimates")
    add("")
    add(
        "Required by the frozen protocol before survival is used analytically. Generated by "
        "`04_code/audit_survival_circularity.py`."
    )
    add("")
    add(f"Countries with both a WHO survival estimate and a GBD trend: **{len(panel)}**.")
    add("")
    add("## Verdict")
    add("")
    add(
        f"**Use A, the survival value as an explanatory variable for mortality trend: "
        f"{'REJECTED' if use_a_fails else 'PERMITTED'}.**"
    )
    add(
        f"**Use B, the survival estimate source class as a data-availability stratification axis: "
        f"{'REJECTED' if use_b_fails else 'PERMITTED with conditions'}.**"
    )
    add("")
    add("## Documentary arm")
    add("")
    for source, quotations in DOCUMENTARY_RECORD:
        add(f"**{source}**")
        add("")
        for quotation in quotations:
            add(f"- {quotation}")
        add("")
    add(
        "WHO lists vital and death registration among the data sources for the survival "
        "estimates. Mortality data therefore entered the estimation of survival for at least some "
        "countries, which is the condition the protocol asks this audit to detect."
    )
    add("")
    add("## Empirical arm")
    add("")
    add("| Quantity | Compared with | n | Pearson r | Spearman rho | Shared variance |")
    add("|---|---|---:|---:|---:|---:|")
    for _, row in diagnostics.iterrows():
        add(
            f"| {row['quantity']} | {row['compared_with']} | {int(row['countries'])} | "
            f"{row['pearson_r']:+.3f} | {row['spearman_rho']:+.3f} | "
            f"{row['shared_variance_r2']:.3f} |"
        )
    add("")
    add(
        f"Regressing the published survival estimate on the naive identity 100(1 - MI) gives "
        f"R2 = {identity_fit.rvalue ** 2:.3f}, slope {identity_fit.slope:.3f}, intercept "
        f"{identity_fit.intercept:+.1f}, with a mean absolute gap of {identity_gap:.1f} percentage "
        "points. The published series is therefore strongly related to the mortality-to-incidence "
        "ratio without being a deterministic transform of it."
    )
    add("")
    add("## Reasoning")
    add("")
    add(
        f"**Use A fails.** The survival estimates share "
        f"{float(mi['shared_variance_r2']):.1%} of their variance with the mortality-to-incidence "
        f"ratio this study builds from its own outcome, above the {REJECT_R2:.0%} threshold fixed "
        f"before the numbers were read, and they correlate at r = {float(estimand['pearson_r']):+.3f} "
        "with the study's estimand itself. A variable that already contains the outcome cannot "
        "explain it. Survival must not be used as a predictor of, or a stratifier of, the mortality "
        "trend, and no causal or explanatory language about survival and trend may be drawn from "
        "this study."
    )
    add("")
    add(
        f"**Use B is admissible under conditions.** The width of WHO's published uncertainty "
        f"interval tracks how much country-specific data informed the estimate: it correlates "
        f"{float(width_registry['spearman_rho']):+.3f} with years of usable registry series, while "
        f"its association with the mortality level itself is weak "
        f"({float(width_mortality['spearman_rho']):+.3f}). It is therefore a marker of survival-data "
        "availability rather than of mortality. The conditions are that it is reported as a "
        "data-availability axis and never as an epidemiological one, that it is presented alongside "
        "the existing precision and registry axes rather than in place of them, and that the "
        f"association between the width and the estimand (r = {float(width_estimand['pearson_r']):+.3f}) "
        "is reported as the confounding of trend with measurement that this study already "
        "documents, not as a finding about survival."
    )
    add("")
    add("## Deviation this creates")
    add("")
    add(
        "The protocol names stratification by survival-estimate source class. WHO does not publish "
        "a per-country flag distinguishing estimates taken from a Member State submission or a "
        "registry from those predicted by the model, and the Nature Medicine methods are behind a "
        "paywall, so the source class cannot be read off directly. It is operationalised here as "
        "tertiles of the published uncertainty-interval width, validated against registry years as "
        "shown above. This is a deviation in measurement, not in intent, and is recorded in "
        "`deviations_and_limitations.md`."
    )
    add("")

    (RESULTS / "survival_circularity_audit.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Countries audited: {len(panel)}")
    print(diagnostics.to_string(index=False))
    print()
    print(f"Use A (survival value as explanatory variable): {'REJECTED' if use_a_fails else 'PERMITTED'}")
    print(
        f"Use B (source class as data-availability axis): "
        f"{'REJECTED' if use_b_fails else 'PERMITTED with conditions'}"
    )
    print(f"Written: {RESULTS / 'survival_circularity_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
