import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation import _binary_metrics, _classification_metrics, _scenario_costs, evaluate
from app.cli import main
from app.llm import MockProvider

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_cost_scenarios_use_assumptions_and_published_rates(self):
        pricing = json.loads((ROOT / "data" / "model_pricing.json").read_text(encoding="utf-8"))
        result = _scenario_costs(pricing)
        self.assertEqual(result["scenarios"]["low"]["estimated_cost_per_1000_tickets_usd"], 0.15)
        self.assertEqual(result["scenarios"]["base"]["estimated_cost_per_1000_tickets_usd"], 0.30)
        self.assertEqual(result["scenarios"]["high"]["estimated_cost_per_1000_tickets_usd"], 0.60)

    def test_classification_metrics_report_macro_f1_and_confusion_matrix(self):
        result = _classification_metrics(["a", "a", "b"], ["a", "b", "b"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["a"]["f1"], 2 / 3)
        self.assertEqual(result["confusion_matrix"]["a"]["b"], 1)

    def test_escalation_metrics_count_false_negatives(self):
        result = _binary_metrics([True, True, False], [True, False, True])
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["confusion_matrix"]["false_positive"], 1)

    def test_mock_evaluation_reports_sample_and_unmeasured_usage(self):
        report = evaluate(
            dataset_path=ROOT / "data" / "evaluation.jsonl",
            policy_path=ROOT / "data" / "policies.jsonl",
            provider=MockProvider(),
            groundedness_path=ROOT / "data" / "groundedness_reviews.jsonl",
        )
        self.assertEqual(report["sample_count"], 10)
        self.assertEqual(report["schema_validity"]["valid_outputs"], 10)
        self.assertIsNone(report["operations"]["input_tokens_observed"])
        self.assertEqual(report["groundedness"]["status"], "not_adjudicated")
        self.assertEqual(report["escalation"]["false_negatives"], 0)
        self.assertGreater(report["intent"]["accuracy"], 0)
        self.assertEqual(report["schema_validity"]["validity_fraction"], 1.0)
        json.dumps(report)

    def test_groundedness_review_requires_current_draft_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            reviews = Path(folder) / "reviews.jsonl"
            reviews.write_text(json.dumps({
                "ticket_id": "E-2001", "draft_sha256": "stale", "reviewer": "reviewer-1",
                "claims": [{"claim": "A factual statement", "disposition": "supported"}],
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "draft hash"):
                evaluate(
                    dataset_path=ROOT / "data" / "evaluation.jsonl",
                    policy_path=ROOT / "data" / "policies.jsonl",
                    provider=MockProvider(),
                    groundedness_path=reviews,
                )

    def test_evaluate_cli_writes_cost_scenarios_to_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "m4.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([
                    "evaluate", "--provider", "mock", "--pricing", str(ROOT / "data" / "model_pricing.json"),
                    "--output", str(report_path),
                ])
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(
            report["operations"]["pricing_scenario_estimates"]["scenarios"]["base"]["estimated_cost_per_1000_tickets_usd"],
            0.30,
        )


if __name__ == "__main__":
    unittest.main()
