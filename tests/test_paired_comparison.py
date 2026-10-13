"""Generated arithmetic/contract checks; no data, models or training."""
import copy
import math
import unittest

from hydra_correctors.evaluation import ForecastKey, HOUR, score_paired
from hydra_correctors.paired_comparison import CONTEXT_SCHEMA, compare_paired


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.keys = [ForecastKey("g", 0, 1), ForecastKey("g", HOUR, 1),
                     ForecastKey("g", 2 * HOUR, 1), ForecastKey("g", 0, 2),
                     ForecastKey("h", 0, 1)]
        self.observations = dict(zip(self.keys, [10., 10., None, 10., 10.]))
        self.baseline = dict(zip(self.keys, [12., 12., 99., 16., 14.]))
        self.candidate = dict(zip(self.keys, [14., 14., 100., 12., 11.]))
        self.scales = {"g": 2., "h": 1.}

    def compare(self, **overrides):
        inputs = dict(expected_keys=self.keys, observations=self.observations,
                      baseline_predictions=self.baseline,
                      candidate_predictions=self.candidate, gauge_scales=self.scales,
                      baseline_id="synthetic-reference", candidate_id="synthetic-candidate")
        inputs.update(overrides)
        return compare_paired(**inputs)

    def test_shared_scores_and_deltas_preserve_gauge_then_lead_weighting(self):
        result = self.compare()
        self.assertEqual(result["status"], "COMPARED_FIXED_SUPPORT")
        self.assertEqual(result["delta_convention"], "candidate_minus_baseline")
        # Unequal row counts and lead counts; one cell worsens as others improve.
        self.assertEqual(result["baseline"]["primary"], 3.)
        self.assertEqual(result["candidate"]["primary"], 1.25)
        self.assertEqual(result["primary_delta"], -1.75)
        self.assertEqual(result["cell_deltas"], [
            {"gauge": "g", "lead": 1, "count": 2, "rmse_delta": 2.,
             "mae_delta": 2., "nrmse_delta": 1.},
            {"gauge": "g", "lead": 2, "count": 1, "rmse_delta": -4.,
             "mae_delta": -4., "nrmse_delta": -2.},
            {"gauge": "h", "lead": 1, "count": 1, "rmse_delta": -3.,
             "mae_delta": -3., "nrmse_delta": -3.},
        ])
        for role, predictions in (("baseline", self.baseline), ("candidate", self.candidate)):
            self.assertEqual(result[role], score_paired(
                self.keys, self.observations, predictions, self.scales))
            self.assertEqual(result[role]["registered_rows"], 5)
            self.assertEqual(result[role]["scored_rows"], 4)
            self.assertEqual(result[role]["missing_target_rows"], 1)

    def test_swapping_forecasts_negates_deltas_and_preserves_context(self):
        forward = self.compare()
        reverse = self.compare(baseline_predictions=self.candidate,
                               candidate_predictions=self.baseline,
                               baseline_id="synthetic-candidate", candidate_id="synthetic-reference")
        self.assertEqual(reverse["context_sha256"], forward["context_sha256"])
        self.assertEqual(reverse["primary_delta"], -forward["primary_delta"])
        for before, after in zip(forward["cell_deltas"], reverse["cell_deltas"], strict=True):
            for metric in ("rmse_delta", "mae_delta", "nrmse_delta"):
                self.assertEqual(after[metric], -before[metric])

    def test_order_and_key_iterator_do_not_change_context(self):
        expected = self.compare()
        actual = self.compare(expected_keys=iter(reversed(self.keys)),
                              observations=dict(reversed(list(self.observations.items()))),
                              baseline_predictions=dict(reversed(list(self.baseline.items()))),
                              candidate_predictions=dict(reversed(list(self.candidate.items()))),
                              gauge_scales={"h": 1., "g": 2.})
        self.assertEqual(actual, expected)
        self.assertEqual(actual["context_schema"], CONTEXT_SCHEMA)
        self.assertRegex(actual["context_sha256"], r"^[0-9a-f]{64}$")

    def test_targets_masks_and_scales_are_bound_by_context(self):
        original = self.compare()["context_sha256"]
        variants = [dict(observations={**self.observations, self.keys[0]: 11.}),
                    dict(observations={**self.observations, self.keys[2]: 10.}),
                    dict(gauge_scales={"g": 3., "h": 1.})]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(self.compare(**variant)["context_sha256"], original)

    def test_keys_are_bound_even_when_values_and_metrics_match(self):
        changed_keys = [ForecastKey(key.gauge, key.issue_epoch + 10 * HOUR, key.lead)
                        for key in self.keys]
        result = self.compare(expected_keys=changed_keys,
                              observations=dict(zip(changed_keys, self.observations.values())),
                              baseline_predictions=dict(zip(changed_keys, self.baseline.values())),
                              candidate_predictions=dict(zip(changed_keys, self.candidate.values())))
        original = self.compare()
        self.assertEqual(result["primary_delta"], original["primary_delta"])
        self.assertNotEqual(result["context_sha256"], original["context_sha256"])

    def test_predictions_and_caller_ids_are_not_context_identity(self):
        result = self.compare(candidate_predictions=self.baseline,
                              baseline_id="other-reference", candidate_id="other-candidate")
        self.assertEqual(result["context_sha256"], self.compare()["context_sha256"])
        self.assertEqual(result["primary_delta"], 0.)

    def test_integer_and_float_targets_and_scales_share_context(self):
        observations = {key: None if value is None else int(value)
                        for key, value in self.observations.items()}
        self.assertEqual(self.compare(observations=observations, gauge_scales={"g": 2, "h": 1}),
                         self.compare())

    def test_missing_or_extra_forecasts_fail_for_either_role(self):
        for role in ("baseline_predictions", "candidate_predictions"):
            missing = dict(self.baseline)
            del missing[self.keys[2]]  # Even a masked target needs its forecast.
            extra = {**self.baseline, ForecastKey("g", 5 * HOUR, 1): 1.}
            for predictions in (missing, extra):
                with self.subTest(role=role, keys=len(predictions)):
                    with self.assertRaisesRegex(ValueError, "exactly match frozen keys"):
                        self.compare(**{role: predictions})

    def test_nonfinite_or_boolean_forecasts_on_masked_rows_fail_for_either_role(self):
        for role in ("baseline_predictions", "candidate_predictions"):
            for value in (math.nan, math.inf, -math.inf, True):
                with self.subTest(role=role, value=value):
                    with self.assertRaisesRegex(ValueError, "finite numeric data"):
                        self.compare(**{role: {**self.baseline, self.keys[2]: value}})

    def test_invalid_shared_context_still_fails(self):
        variants = [dict(expected_keys=[]), dict(expected_keys=self.keys + self.keys),
                    dict(expected_keys=["not-a-forecast-key"]),
                    dict(observations={**self.observations, self.keys[-1]: None}),
                    dict(observations={**self.observations, self.keys[0]: math.nan}),
                    dict(gauge_scales={"g": 0., "h": 1.}), dict(gauge_scales={"g": 2.})]
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    self.compare(**variant)

    def test_nonmapping_inputs_do_not_silently_collapse_pairs(self):
        for name, values in (("observations", self.observations),
                             ("baseline_predictions", self.baseline),
                             ("candidate_predictions", self.candidate),
                             ("gauge_scales", self.scales)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "must be a mapping"):
                    self.compare(**{name: list(values.items())})

    def test_nonblank_distinct_caller_ids_required(self):
        for value in (None, "", "  ", True):
            for role in ("baseline_id", "candidate_id"):
                with self.subTest(role=role, value=value):
                    with self.assertRaises(ValueError):
                        self.compare(**{role: value})
        with self.assertRaisesRegex(ValueError, "identities must differ"):
            self.compare(baseline_id="same", candidate_id="same")

    def test_zero_reference_error_and_undefined_bias_remain_valid(self):
        zeros = dict.fromkeys(self.keys, 0.)
        result = self.compare(observations=zeros, baseline_predictions=zeros,
                              candidate_predictions=dict.fromkeys(self.keys, 1.))
        self.assertEqual(result["baseline"]["primary"], 0.)
        self.assertEqual(result["primary_delta"], .75)
        for role in ("baseline", "candidate"):
            for cell in result[role]["cells"]:
                self.assertIsNone(cell["pbias_percent"])
                self.assertEqual(cell["pbias_undefined"], "zero_observation_sum")

    def test_caller_inputs_are_unchanged(self):
        inputs = (self.keys, self.observations, self.baseline, self.candidate, self.scales)
        before = copy.deepcopy(inputs)
        self.compare()
        self.assertEqual(inputs, before)


if __name__ == "__main__":
    unittest.main()
