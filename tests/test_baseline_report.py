from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from hydra_correctors.baseline_report import read_metrics, validate_bundle, validate_metrics


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/c_physical_baseline_2019_10"


class BaselineReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_metrics(REPORT / "metrics_by_lead.csv")
        self.summary = json.loads((REPORT / "summary.json").read_text())

    def test_retained_evidence_reproduces_exact_scope_and_pooled_metrics(self):
        result = validate_bundle(REPORT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["pooled_recomputed"]["cells"], 13203)
        self.assertAlmostEqual(result["pooled_recomputed"]["RMSE_cms"], 11.686679365777943, places=12)
        self.assertAlmostEqual(result["pooled_recomputed"]["MAE_cms"], 6.59026054242611, places=12)
        self.assertAlmostEqual(result["pooled_recomputed"]["bias_forecast_minus_observation_cms"], 5.972715863770979, places=12)

    def test_missing_duplicate_or_reordered_lead_is_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[-1]], list(reversed(self.rows))]:
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                validate_metrics(rows, self.summary)

    def test_inconsistent_support_and_statistics_are_rejected(self):
        changes = [("scored_cells", 741), ("SSE_cms_squared", -1.0), ("SAE_cms", float("nan")), ("RMSE_cms", 0.0), ("signed_error_sum_cms", 999999.0), ("normalized_RMSE", float("inf")), ("raw_forecast_cells", True)]
        for key, value in changes:
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_metrics(rows, self.summary)

    def test_changed_window_or_unsupported_comparison_cannot_promote_result(self):
        for key, value in [("forecast_issues", 745), ("learned_fits", 1), ("learned_improvement_claim", True), ("paired_corrected_comparison_available", True)]:
            summary = deepcopy(self.summary)
            summary[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_metrics(self.rows, summary)
        summary = deepcopy(self.summary)
        summary["issue_window_UTC"]["stop_exclusive"] = "2019-11-02T00:00:00Z"
        with self.assertRaises(ValueError):
            validate_metrics(self.rows, summary)

    def test_changed_report_bytes_are_rejected_before_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ["metrics_by_lead.csv", "summary.json", "run_manifest.json", "provenance.json"]:
                (directory / name).write_bytes((REPORT / name).read_bytes())
            (directory / "metrics_by_lead.csv").write_text("not the retained result\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_bundle(directory)

    def test_pool_support_or_fixed_scale_drift_is_rejected(self):
        for field in ["raw_cells", "scored_cells", "excluded_cells"]:
            summary = deepcopy(self.summary)
            summary[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_metrics(self.rows, summary)
        summary = deepcopy(self.summary)
        summary["normalization"]["fixed_reference_scale_cms"] = 0.0
        with self.assertRaises(ValueError):
            validate_metrics(self.rows, summary)


if __name__ == "__main__":
    unittest.main()
