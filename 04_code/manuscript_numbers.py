"""Print every quantity the manuscript needs, read from the machine-readable results.

Written so that no number in the manuscript is typed from memory or recomputed by hand. Run from
the project root with the project virtual environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    section("DIAGNOSTIC CLASSIFICATION")
    diagnostic = pd.read_csv(RESULTS / "primary_country_trajectory_diagnostic.csv")
    print("countries:", len(diagnostic))
    print(diagnostic["trajectory_class"].value_counts().to_dict())
    print("median APC: %.4f" % diagnostic["apc_point"].median())
    print("IQR APC: %.4f to %.4f" % (diagnostic["apc_point"].quantile(0.25), diagnostic["apc_point"].quantile(0.75)))
    print("range APC: %.4f to %.4f" % (diagnostic["apc_point"].min(), diagnostic["apc_point"].max()))
    print("countries with declining point estimate:", int((diagnostic["apc_point"] < 0).sum()))
    print("years used, min/median/max:", diagnostic["n_years"].min(), diagnostic["n_years"].median(), diagnostic["n_years"].max())
    print("on trajectory:", diagnostic.loc[diagnostic["trajectory_class"].eq("on_trajectory"), "location"].tolist())
    print("uncertain:", diagnostic.loc[diagnostic["trajectory_class"].eq("uncertain"), "location"].tolist())

    section("CONFIRMATORY CLASSIFICATION (World Bank region pooling)")
    hierarchical = pd.read_csv(RESULTS / "hierarchical_country_trajectory.csv")
    print("countries:", len(hierarchical))
    print(hierarchical["trajectory_class"].value_counts().to_dict())
    print("on trajectory:", hierarchical.loc[hierarchical["trajectory_class"].eq("on_trajectory"), ["location", "apc_pooled", "p_apc_le_minus_2_5"]].to_string(index=False))
    print("uncertain:", hierarchical.loc[hierarchical["trajectory_class"].eq("uncertain"), ["location", "apc_pooled", "p_apc_le_minus_2_5"]].to_string(index=False))
    merged = hierarchical.merge(diagnostic[["location", "trajectory_class", "apc_point"]], on="location", suffixes=("_conf", "_diag"))
    print("correlation unpooled vs diagnostic APC: %.6f" % merged["apc_unpooled"].corr(merged["apc_point"]))
    changed = merged[merged["trajectory_class_conf"] != merged["trajectory_class_diag"]]
    print("class changes:", len(changed))
    print(changed[["location", "trajectory_class_diag", "trajectory_class_conf"]].to_string(index=False))
    print("median shrinkage to group: %.4f" % hierarchical["shrinkage_to_group"].median())

    section("CONFIRMATORY CLASSIFICATION (WHO region pooling)")
    who_pooled = pd.read_csv(RESULTS / "hierarchical_country_trajectory_who_region.csv")
    print(who_pooled["trajectory_class"].value_counts().to_dict())
    both = who_pooled.merge(hierarchical[["location", "trajectory_class"]], on="location", suffixes=("_who", "_wb"))
    print("countries classified differently by the two taxonomies:", int((both["trajectory_class_who"] != both["trajectory_class_wb"]).sum()))
    print("correlation of pooled APC between taxonomies: %.6f" % both.merge(who_pooled[["location", "apc_pooled"]], on="location")["apc_pooled_x"].corr(both.merge(hierarchical[["location", "apc_pooled"]], on="location")["apc_pooled_y"]) if False else "")

    section("SUBGROUP STRATIFICATION")
    summary = pd.read_csv(RESULTS / "subgroup_stratification.csv")
    for axis in summary["axis"].unique():
        block = summary[summary["axis"].eq(axis)]
        print(f"\n-- {axis}")
        print(block[["level", "countries", "apc_mean", "apc_lower", "apc_upper", "median_apc", "on_trajectory", "uncertain", "off_trajectory", "i_squared_percent"]].to_string(index=False))

    section("BETWEEN-STRATUM HETEROGENEITY")
    print(pd.read_csv(RESULTS / "subgroup_heterogeneity.csv").to_string(index=False))

    section("WHO REGION AGGREGATE CROSS-CHECK")
    print(pd.read_csv(RESULTS / "who_region_aggregate_validation.csv").to_string(index=False))

    section("SUBGROUP ASSIGNMENTS: COVERAGE OF EACH AXIS")
    assignments = pd.read_csv(RESULTS / "subgroup_country_assignments.csv")
    print("countries:", len(assignments))
    for column in ("who_region", "income_group", "registry_series"):
        print(f"\n{column}:")
        print(assignments[column].value_counts().to_string())
    print("\nunassigned WHO region countries:", assignments.loc[assignments["who_region"].str.startswith("Not assigned"), "location"].tolist())
    print("registry years, countries with any:", int((assignments["registry_years_in_window"] > 0).sum()))

    section("DECOMPOSITION")
    print(pd.read_csv(RESULTS / "decomposition_summary.csv").to_string(index=False))

    section("FORECAST SCENARIOS")
    forecast = pd.read_csv(RESULTS / "forecast_scenario_summary.csv")
    print(forecast.to_string(index=False))
    backtest = pd.read_csv(RESULTS / "forecast_backtest.csv")
    print("\nback-test columns:", backtest.columns.tolist())
    error = backtest.filter(like="error").columns
    print(backtest[error].describe().to_string() if len(error) else backtest.head().to_string())

    section("HEALTH SYSTEM DISTRIBUTED LAG")
    print(pd.read_csv(RESULTS / "health_system_lag_summary.csv").to_string(index=False))
    print()
    print(pd.read_csv(RESULTS / "health_system_lag_coefficients.csv").to_string(index=False))

    section("CONSTRUCTED INCIDENCE VALIDATION")
    validation = pd.read_csv(RESULTS / "constructed_incidence_validation.csv")
    print("rows:", len(validation), "columns:", validation.columns.tolist())
    numeric = validation.select_dtypes("number")
    if {"constructed_asr", "gbd_asr"}.issubset(set(validation.columns)):
        print("pearson r: %.6f" % validation["constructed_asr"].corr(validation["gbd_asr"]))
    print(numeric.describe().to_string())

    section("EXECUTION STATUS")
    print(json.dumps(json.loads((RESULTS / "execution_status.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
