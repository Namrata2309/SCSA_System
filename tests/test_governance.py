import unittest

from app.governance import route_failure, route_ticket
from app.schemas import ExtractedEntities, ModelAssessment, PolicyEvidence


def assessment(**overrides):
    values = {
        "category": "shipping",
        "sentiment": "neutral",
        "confidence_score": 0.75,
        "extracted_entities": ExtractedEntities(disputed_amount=100),
    }
    values.update(overrides)
    return ModelAssessment(**values)


def evidence(score=0.60):
    return [PolicyEvidence(
        policy_id="POL-1", category="shipping", excerpt="Shipping policy.",
        similarity_score=score,
    )]


class GovernanceTests(unittest.TestCase):
    def test_threshold_equalities_do_not_trigger(self):
        result = route_ticket(assessment(), evidence())
        self.assertFalse(result.human_in_loop_required)
        self.assertTrue(result.automation_eligible)
        self.assertEqual(result.escalation_reason, [])

    def test_strict_numeric_thresholds_trigger(self):
        result = route_ticket(
            assessment(confidence_score=0.749, extracted_entities=ExtractedEntities(disputed_amount=100.01)),
            evidence(0.599),
        )
        self.assertEqual(
            result.escalation_reason,
            ["low_confidence", "high_disputed_amount", "low_policy_similarity"],
        )

    def test_confidence_below_equal_and_above_boundary(self):
        cases = ((0.749, True), (0.75, False), (0.751, False))
        for confidence, should_review in cases:
            with self.subTest(confidence=confidence):
                decision = route_ticket(assessment(confidence_score=confidence), evidence())
                self.assertEqual(decision.human_in_loop_required, should_review)

    def test_amount_below_equal_and_above_boundary(self):
        cases = ((99.99, False), (100.0, False), (100.01, True))
        for amount, should_review in cases:
            with self.subTest(amount=amount):
                decision = route_ticket(
                    assessment(extracted_entities=ExtractedEntities(disputed_amount=amount)),
                    evidence(),
                )
                self.assertEqual(decision.human_in_loop_required, should_review)

    def test_similarity_below_equal_and_above_boundary(self):
        cases = ((0.599, True), (0.60, False), (0.601, False))
        for similarity, should_review in cases:
            with self.subTest(similarity=similarity):
                decision = route_ticket(assessment(), evidence(similarity))
                self.assertEqual(decision.human_in_loop_required, should_review)

    def test_each_risk_trigger_is_retained(self):
        result = route_ticket(
            assessment(
                confidence_score=0.7,
                legal_threat=True,
                sentiment="extreme_frustration",
                extracted_entities=ExtractedEntities(disputed_amount=101),
            ),
            evidence(0.2),
        )
        self.assertEqual(
            result.escalation_reason,
            ["low_confidence", "high_disputed_amount", "low_policy_similarity", "legal_threat", "extreme_frustration"],
        )

    def test_missing_evidence_and_amount(self):
        result = route_ticket(assessment(extracted_entities=ExtractedEntities()), [])
        self.assertEqual(result.escalation_reason, ["missing_policy_evidence"])

    def test_failure_routes_to_review(self):
        result = route_failure("provider_timeout")
        self.assertTrue(result.human_in_loop_required)
        self.assertFalse(result.automation_eligible)
        self.assertEqual(result.escalation_reason, ["provider_timeout"])


if __name__ == "__main__":
    unittest.main()
