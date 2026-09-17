import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import (
    GBD_AGE_BAND_WEIGHTS,
    GBD_NAME_TO_ISO3,
    MIN_YEARS_PRIMARY,
    NO_SURVIVAL_ESTIMATE,
    SUBGROUP_AXES,
    SURVIVAL_SOURCE_CLASS_LABELS,
    UNASSIGNED_WHO_REGION,
    WHO_REGION_LOCATION_IDS,
    WHO_REGION_NAMES,
    WHO_STANDARD_WEIGHTS,
    WPP_TO_GBD_BAND,
    _between_stratum_test,
    _country_slope_with_measurement_error,
    _fit_group_hyperparameters,
    _logarithmic_mean,
    _random_effects_summary,
    _tertile_labels,
    apc_from_log_slope,
    baseline_and_precision,
    classify_probability,
    interval_to_log_sd,
    is_who_region_aggregate,
    monte_carlo_country_trend,
    validate_primary,
    who_region_crosswalk,
    who_survival_source_class,
)


class AggregateSeparationTests(unittest.TestCase):
    """WHO regional aggregates must never enter a country-level trajectory analysis."""

    def _frame(self):
        return pd.DataFrame(
            {
                "location": [
                    "Japan",
                    "African Region",
                    "Region of the Americas",
                    "Nigeria",
                ],
                "location_id": [67, 44563, 44564, 214],
                "year": [2020, 2020, 2020, 2020],
                "rate": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def test_regions_flagged_and_countries_kept(self):
        flags = is_who_region_aggregate(self._frame())
        self.assertEqual(list(flags), [False, True, True, False])

    def test_detection_survives_missing_location_id(self):
        frame = self._frame().drop(columns=["location_id"])
        flags = is_who_region_aggregate(frame)
        self.assertEqual(list(flags), [False, True, True, False])

    def test_detection_survives_renamed_region(self):
        """A translated or reworded region label is still caught by its numeric id."""
        frame = self._frame()
        frame.loc[1, "location"] = "Region africaine"
        self.assertTrue(bool(is_who_region_aggregate(frame).iloc[1]))

    def test_bare_who_region_label_is_excluded(self):
        """Location 479 is exported as the bare label 'WHO region' and is not a country."""
        frame = pd.DataFrame(
            {"location": ["WHO region", "Japan"], "location_id": [479, 67], "year": [2000, 2000]}
        )
        self.assertEqual(list(is_who_region_aggregate(frame)), [True, False])

    def test_registered_aggregate_ids(self):
        self.assertEqual(len(WHO_REGION_LOCATION_IDS), 7)


class AgeStandardisationTests(unittest.TestCase):
    """The constructed incidence series must reproduce direct standardisation exactly."""

    def test_band_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(GBD_AGE_BAND_WEIGHTS.values()), 1.0, places=10)

    def test_under_twenty_band_carries_four_standard_groups(self):
        expected = float(WHO_STANDARD_WEIGHTS[0:4].sum())
        self.assertAlmostEqual(GBD_AGE_BAND_WEIGHTS["<20 years"], expected, places=12)

    def test_constant_rate_standardises_to_itself(self):
        """A population with the same rate in every band must standardise to that rate."""
        total = sum(12.5 * weight for weight in GBD_AGE_BAND_WEIGHTS.values())
        self.assertAlmostEqual(total, 12.5, places=10)

    def test_weights_match_gbd_band_labels(self):
        self.assertEqual(len(GBD_AGE_BAND_WEIGHTS), 15)
        self.assertIn("85+ years", GBD_AGE_BAND_WEIGHTS)
        self.assertNotIn("85-89 years", GBD_AGE_BAND_WEIGHTS)


class WeightedSlopeTests(unittest.TestCase):
    """Stage one of the hierarchical model: weighted slope with measurement variance."""

    def _frame(self, slope=-0.02, shift=0.0, year_shift=0, widths=None):
        years = np.arange(2015, 2024) + year_shift
        rate = np.exp(shift + slope * (years - years[0]) + math.log(20.0))
        widths = widths if widths is not None else np.full(len(years), 0.10)
        return pd.DataFrame(
            {
                "year": years,
                "rate": rate,
                "lower": rate * (1 - widths),
                "upper": rate * (1 + widths),
            }
        )

    def test_recovers_known_slope(self):
        result = _country_slope_with_measurement_error(self._frame(slope=-0.02))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], -0.02, places=9)

    def test_slope_invariant_to_level_shift(self):
        """Scaling every rate by a constant must not move the slope.

        The original implementation centred the year on the unweighted mean and left the outcome
        uncentred, so a level shift leaked into the slope. This test pins that fix.
        """
        base = _country_slope_with_measurement_error(self._frame(slope=-0.03))
        shifted = _country_slope_with_measurement_error(self._frame(slope=-0.03, shift=2.5))
        self.assertAlmostEqual(base[0], shifted[0], places=9)

    def test_slope_invariant_to_year_offset(self):
        base = _country_slope_with_measurement_error(self._frame(slope=-0.03))
        offset = _country_slope_with_measurement_error(self._frame(slope=-0.03, year_shift=50))
        self.assertAlmostEqual(base[0], offset[0], places=9)

    def test_unequal_weights_still_recover_slope(self):
        widths = np.linspace(0.05, 0.30, 9)
        result = _country_slope_with_measurement_error(self._frame(slope=-0.025, widths=widths))
        self.assertAlmostEqual(result[0], -0.025, places=9)

    def test_short_series_rejected(self):
        short = self._frame().head(MIN_YEARS_PRIMARY - 1)
        self.assertIsNone(_country_slope_with_measurement_error(short))


class PartialPoolingTests(unittest.TestCase):
    def test_identical_countries_give_zero_heterogeneity(self):
        slopes = np.full(8, -0.02)
        variances = np.full(8, 1e-4)
        mean, tau = _fit_group_hyperparameters(slopes, variances)
        self.assertAlmostEqual(mean, -0.02, places=6)
        self.assertEqual(tau, 0.0)

    def test_spread_countries_give_positive_heterogeneity(self):
        slopes = np.array([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04])
        variances = np.full(6, 1e-6)
        _, tau = _fit_group_hyperparameters(slopes, variances)
        self.assertGreater(tau, 0.01)

    def test_single_country_group_is_not_shrunk_away(self):
        mean, tau = _fit_group_hyperparameters(np.array([-0.05]), np.array([1e-4]))
        self.assertAlmostEqual(mean, -0.05, places=9)
        self.assertEqual(tau, 0.0)

    def test_group_mean_is_precision_weighted(self):
        """A precise country should pull the group mean more than a noisy one."""
        slopes = np.array([-0.05, 0.05])
        mean, _ = _fit_group_hyperparameters(slopes, np.array([1e-6, 1e-2]))
        self.assertLess(mean, 0.0)


class LogarithmicMeanTests(unittest.TestCase):
    """The LMDI weight must behave correctly, including at equal values."""

    def test_equal_values_return_the_value(self):
        result = _logarithmic_mean(np.array([4.0]), np.array([4.0]))
        self.assertAlmostEqual(float(result[0]), 4.0, places=12)

    def test_lies_between_the_two_values(self):
        result = float(_logarithmic_mean(np.array([1.0]), np.array([100.0]))[0])
        self.assertGreater(result, 1.0)
        self.assertLess(result, 100.0)

    def test_symmetric(self):
        forward = float(_logarithmic_mean(np.array([3.0]), np.array([7.0]))[0])
        backward = float(_logarithmic_mean(np.array([7.0]), np.array([3.0]))[0])
        self.assertAlmostEqual(forward, backward, places=12)

    def test_zero_or_negative_inputs_give_zero_weight(self):
        result = _logarithmic_mean(np.array([0.0, -1.0]), np.array([5.0, 5.0]))
        self.assertEqual(list(result), [0.0, 0.0])

    def test_decomposition_is_exact_on_a_synthetic_case(self):
        """Four multiplicative factors must decompose with no residual."""
        pop0, pop1 = np.array([1000.0, 2000.0]), np.array([1100.0, 2400.0])
        share0, share1 = np.array([0.4, 0.6]), np.array([0.35, 0.65])
        inc0, inc1 = np.array([10.0, 40.0]), np.array([12.0, 44.0])
        mir0, mir1 = np.array([0.5, 0.6]), np.array([0.45, 0.55])
        deaths0 = pop0 * share0 * inc0 * mir0
        deaths1 = pop1 * share1 * inc1 * mir1
        weight = _logarithmic_mean(deaths1, deaths0)
        total = (
            weight * np.log(pop1 / pop0)
            + weight * np.log(share1 / share0)
            + weight * np.log(inc1 / inc0)
            + weight * np.log(mir1 / mir0)
        )
        self.assertAlmostEqual(float(total.sum()), float(deaths1.sum() - deaths0.sum()), places=6)


class PopulationBandTests(unittest.TestCase):
    """WPP bands must collapse onto exactly the GBD bands, losing no population."""

    def test_every_wpp_band_maps(self):
        expected = (
            ["0-4", "5-9", "10-14", "15-19"]
            + [f"{s}-{s + 4}" for s in range(20, 85, 5)]
            + ["85-89", "90-94", "95-99", "100+"]
        )
        self.assertEqual(sorted(WPP_TO_GBD_BAND), sorted(expected))

    def test_mapping_targets_are_exactly_the_gbd_bands(self):
        self.assertEqual(set(WPP_TO_GBD_BAND.values()), set(GBD_AGE_BAND_WEIGHTS))

    def test_under_twenty_and_open_ended_bands_are_collapsed(self):
        self.assertEqual(WPP_TO_GBD_BAND["0-4"], "<20 years")
        self.assertEqual(WPP_TO_GBD_BAND["15-19"], "<20 years")
        self.assertEqual(WPP_TO_GBD_BAND["100+"], "85+ years")
        self.assertEqual(WPP_TO_GBD_BAND["85-89"], "85+ years")

    def test_middle_bands_are_one_to_one(self):
        self.assertEqual(WPP_TO_GBD_BAND["50-54"], "50-54 years")
        self.assertEqual(WPP_TO_GBD_BAND["80-84"], "80-84 years")


class CrosswalkTests(unittest.TestCase):
    def test_manual_map_has_no_duplicate_iso3(self):
        values = list(GBD_NAME_TO_ISO3.values())
        self.assertEqual(len(values), len(set(values)), "two GBD names map to the same ISO3")

    def test_manual_map_codes_are_three_letters(self):
        for name, code in GBD_NAME_TO_ISO3.items():
            self.assertRegex(code, r"^[A-Z]{3}$", f"bad ISO3 for {name}")

    def test_known_renames_present(self):
        self.assertEqual(GBD_NAME_TO_ISO3["United States of America"], "USA")
        self.assertEqual(GBD_NAME_TO_ISO3["Türkiye"], "TUR")
        self.assertEqual(GBD_NAME_TO_ISO3["Republic of Korea"], "KOR")
        # Congo and DR Congo are genuinely different countries and must not collide.
        self.assertNotEqual(
            GBD_NAME_TO_ISO3["Congo"], GBD_NAME_TO_ISO3["Democratic Republic of the Congo"]
        )


class FormulaTests(unittest.TestCase):
    def test_minus_2_5_percent_slope(self):
        slope = math.log(0.975)
        self.assertAlmostEqual(apc_from_log_slope(slope), -2.5, places=10)

    def test_probability_classification_boundaries(self):
        self.assertEqual(classify_probability(0.80), "on_trajectory")
        self.assertEqual(classify_probability(0.20), "uncertain")
        self.assertEqual(classify_probability(0.199999), "off_trajectory")

    def test_interval_sd_positive(self):
        sd = interval_to_log_sd(np.array([10.0]), np.array([8.0]), np.array([12.0]))
        self.assertGreater(sd[0], 0)


class TrendTests(unittest.TestCase):
    def frame(self, annual_factor=0.97, years=range(2015, 2024)):
        rates = np.array([20 * annual_factor ** (year - 2015) for year in years])
        return pd.DataFrame(
            {
                "location": "Testland",
                "year": list(years),
                "rate": rates,
                "lower": rates * 0.99,
                "upper": rates * 1.01,
            }
        )

    def test_known_three_percent_decline(self):
        result = monte_carlo_country_trend(self.frame(), draws=1000, seed=1)
        self.assertAlmostEqual(result["apc_point"], -3.0, places=8)
        self.assertGreater(result["p_apc_le_minus_2_5"], 0.95)

    def test_minimum_year_rule(self):
        with self.assertRaises(ValueError):
            monte_carlo_country_trend(self.frame(years=range(2015, 2020)))

    def test_validation_detects_bad_bounds(self):
        frame = self.frame()
        frame.loc[0, "lower"] = frame.loc[0, "rate"] + 1
        self.assertTrue(any("bracket" in item for item in validate_primary(frame)))


class RandomEffectsSummaryTests(unittest.TestCase):
    """The stratum summary must reduce to known closed forms in the cases where one exists."""

    def test_homogeneous_stratum_reduces_to_inverse_variance_mean(self):
        """With no between-country spread the estimator is the fixed-effect mean and its SE."""
        slopes = np.array([-0.02, -0.02, -0.02])
        variances = np.array([1e-4, 4e-4, 9e-4])
        result = _random_effects_summary(slopes, variances)
        expected_se = 1.0 / math.sqrt(float(np.sum(1.0 / variances)))
        self.assertAlmostEqual(result["mean_log_slope"], -0.02, places=10)
        self.assertAlmostEqual(result["between_country_sd_log"], 0.0, places=10)
        self.assertAlmostEqual(result["standard_error_log"], expected_se, places=10)
        self.assertAlmostEqual(result["i_squared_percent"], 0.0, places=10)

    def test_precision_weighting_is_not_a_simple_average(self):
        """A precise country must move the stratum mean further than an imprecise one.

        A plain mean of the two slopes would be zero, so this fails if weighting is dropped.
        """
        result = _random_effects_summary(np.array([-0.04, 0.04]), np.array([1e-6, 1e-2]))
        self.assertLess(result["mean_log_slope"], -0.03)

    def test_dispersed_slopes_produce_high_heterogeneity(self):
        """Slopes far apart relative to their own precision give I squared near 100."""
        slopes = np.array([-0.05, 0.0, 0.05, 0.10])
        variances = np.full(4, 1e-8)
        result = _random_effects_summary(slopes, variances)
        self.assertGreater(result["i_squared_percent"], 99.0)
        self.assertGreater(result["between_country_sd_log"], 0.0)

    def test_apc_bounds_bracket_the_mean(self):
        result = _random_effects_summary(np.array([-0.03, -0.01, -0.02]), np.array([1e-4, 1e-4, 1e-4]))
        self.assertLess(result["apc_lower"], result["apc_mean"])
        self.assertLess(result["apc_mean"], result["apc_upper"])
        self.assertAlmostEqual(result["apc_mean"], apc_from_log_slope(result["mean_log_slope"]), places=12)

    def test_single_country_stratum_does_not_crash(self):
        result = _random_effects_summary(np.array([-0.03]), np.array([1e-4]))
        self.assertEqual(result["countries"], 1)
        self.assertEqual(result["q_degrees_of_freedom"], 0)
        self.assertEqual(result["i_squared_percent"], 0.0)


class BetweenStratumTests(unittest.TestCase):
    def _rows(self, means):
        return [{"mean_log_slope": mean, "standard_error_log": 0.001} for mean in means]

    def test_identical_strata_give_no_evidence_of_difference(self):
        result = _between_stratum_test("axis", "Axis", self._rows([-0.02, -0.02, -0.02]))
        self.assertAlmostEqual(result["q_between"], 0.0, places=12)
        self.assertAlmostEqual(result["p_value"], 1.0, places=10)
        self.assertEqual(result["degrees_of_freedom"], 2)

    def test_separated_strata_are_detected(self):
        result = _between_stratum_test("axis", "Axis", self._rows([-0.03, 0.0, 0.03]))
        self.assertGreater(result["q_between"], 100.0)
        self.assertLess(result["p_value"], 1e-6)

    def test_single_level_axis_returns_no_test(self):
        result = _between_stratum_test("axis", "Axis", self._rows([-0.02]))
        self.assertEqual(result["degrees_of_freedom"], 0)
        self.assertTrue(math.isnan(result["p_value"]))


class TertileTests(unittest.TestCase):
    def test_equal_sized_thirds(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        labels = _tertile_labels(values, ["low", "mid", "high"])
        self.assertEqual(sorted(labels.value_counts().tolist()), [3, 3, 3])
        self.assertEqual(labels.iloc[0], "low")
        self.assertEqual(labels.iloc[-1], "high")

    def test_ties_do_not_empty_a_stratum(self):
        """Ranking before cutting keeps every stratum populated when values repeat.

        Cutting on the raw values would give duplicate bin edges and collapse a stratum.
        """
        values = pd.Series([1.0] * 4 + [2.0] * 3 + [3.0] * 2)
        labels = _tertile_labels(values, ["low", "mid", "high"])
        self.assertEqual(len(set(labels.dropna())), 3)
        self.assertEqual(sorted(labels.value_counts().tolist()), [3, 3, 3])

    def test_two_distinct_values_is_degenerate_not_a_tie(self):
        """A variable with two distinct values has no meaningful tertiles, so none are assigned."""
        labels = _tertile_labels(pd.Series([1.0] * 4 + [2.0] * 5), ["low", "mid", "high"])
        self.assertTrue(labels.isna().all())

    def test_missing_values_stay_missing(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        labels = _tertile_labels(values, ["low", "mid", "high"])
        self.assertTrue(pd.isna(labels.iloc[2]))
        self.assertEqual(labels.notna().sum(), 5)

    def test_too_few_distinct_values_returns_all_missing(self):
        labels = _tertile_labels(pd.Series([5.0, 5.0, 5.0]), ["low", "mid", "high"])
        self.assertTrue(labels.isna().all())


class BaselineAndPrecisionTests(unittest.TestCase):
    def frame(self):
        rows = []
        for location, rate, width in (("Wide", 20.0, 0.5), ("Narrow", 10.0, 0.05)):
            for year in range(2015, 2024):
                rows.append(
                    {
                        "location": location,
                        "year": year,
                        "rate": rate,
                        "lower": rate * (1 - width / 2),
                        "upper": rate * (1 + width / 2),
                    }
                )
        return pd.DataFrame(rows)

    def test_baseline_is_the_first_window_year(self):
        result = baseline_and_precision(self.frame()).set_index("location")
        self.assertAlmostEqual(result.loc["Wide", "baseline_asmr_2015"], 20.0)
        self.assertAlmostEqual(result.loc["Narrow", "baseline_asmr_2015"], 10.0)

    def test_relative_interval_width_is_scale_free(self):
        """A country with a wider interval scores wider even though its rate is higher."""
        result = baseline_and_precision(self.frame()).set_index("location")
        self.assertAlmostEqual(result.loc["Wide", "median_relative_ui_width"], 0.5, places=10)
        self.assertAlmostEqual(result.loc["Narrow", "median_relative_ui_width"], 0.05, places=10)

    def test_years_outside_the_window_are_ignored(self):
        frame = self.frame()
        stray = frame[frame["year"].eq(2015)].copy()
        stray["year"] = 2014
        stray["rate"] = 999.0
        result = baseline_and_precision(pd.concat([frame, stray], ignore_index=True)).set_index("location")
        self.assertAlmostEqual(result.loc["Wide", "baseline_asmr_2015"], 20.0)


class WhoRegionCrosswalkTests(unittest.TestCase):
    """The crosswalk is read from WHO's own ParentLocationCode, so it must match WHO's taxonomy."""

    @classmethod
    def setUpClass(cls):
        cls.crosswalk = who_region_crosswalk()

    def test_only_the_six_who_regions_appear(self):
        if self.crosswalk.empty:
            self.skipTest("no WHO GHO snapshot present")
        self.assertTrue(set(self.crosswalk["who_region_code"]).issubset(set(WHO_REGION_NAMES)))

    def test_one_region_per_country(self):
        if self.crosswalk.empty:
            self.skipTest("no WHO GHO snapshot present")
        self.assertEqual(len(self.crosswalk), self.crosswalk["iso3"].nunique())

    def test_codes_are_iso3(self):
        if self.crosswalk.empty:
            self.skipTest("no WHO GHO snapshot present")
        self.assertTrue(self.crosswalk["iso3"].str.fullmatch(r"[A-Z]{3}").all())

    def test_region_names_are_resolved_not_left_as_codes(self):
        if self.crosswalk.empty:
            self.skipTest("no WHO GHO snapshot present")
        self.assertTrue(set(self.crosswalk["who_region"]).issubset(set(WHO_REGION_NAMES.values())))


class SubgroupAxisTests(unittest.TestCase):
    def test_protocol_axes_are_all_covered(self):
        """Objective 6 names income, region, baseline mortality and data quality."""
        for axis in (
            "income_group",
            "who_region",
            "baseline_mortality_tertile",
            "outcome_precision_tertile",
            "registry_series",
            "survival_source_class",
        ):
            self.assertIn(axis, SUBGROUP_AXES)

    def test_unassigned_label_is_not_a_who_region(self):
        self.assertNotIn(UNASSIGNED_WHO_REGION, set(WHO_REGION_NAMES.values()))


class SurvivalSourceClassTests(unittest.TestCase):
    """The survival axis carries the source class only, never the survival value.

    The predictor-circularity audit rejects the survival value as an explanatory variable because
    it shares most of its variance with the mortality-to-incidence ratio built from this study's
    own outcome. These tests pin that boundary so a later edit cannot quietly cross it.
    """

    def test_source_class_frame_never_exposes_the_survival_value(self):
        frame = who_survival_source_class()
        if frame.empty:
            self.skipTest("no WHO survival snapshot present")
        self.assertEqual(set(frame.columns), {"iso3", "survival_ui_width"})
        self.assertNotIn("survival", frame.columns)
        self.assertNotIn("NumericValue", frame.columns)

    def test_source_class_widths_are_positive_and_unique_per_country(self):
        frame = who_survival_source_class()
        if frame.empty:
            self.skipTest("no WHO survival snapshot present")
        self.assertTrue((frame["survival_ui_width"] > 0).all())
        self.assertEqual(frame["iso3"].nunique(), len(frame))

    def test_missing_survival_gets_its_own_label_rather_than_a_tertile(self):
        """A country with no WHO estimate must not be silently binned with data-rich countries."""
        self.assertNotIn(NO_SURVIVAL_ESTIMATE, SURVIVAL_SOURCE_CLASS_LABELS)

    def test_tertile_labels_are_ordered_from_most_to_least_informed(self):
        self.assertEqual(len(SURVIVAL_SOURCE_CLASS_LABELS), 3)
        self.assertIn("Most", SURVIVAL_SOURCE_CLASS_LABELS[0])
        self.assertIn("Least", SURVIVAL_SOURCE_CLASS_LABELS[-1])


if __name__ == "__main__":
    unittest.main()
