"""Deterministic routing rules, independent of model prompt or provider."""

from .schemas import ModelAssessment, PolicyEvidence, RoutingDecision


CONFIDENCE_THRESHOLD = 0.75
DISPUTED_AMOUNT_THRESHOLD_USD = 100.0
RETRIEVAL_SIMILARITY_THRESHOLD = 0.60


def route_ticket(
    assessment: ModelAssessment,
    evidence: list[PolicyEvidence],
) -> RoutingDecision:
    """Return a review decision when any mandatory trigger or safe fallback applies.

    Comparisons are strict: values exactly at a numeric threshold do not trigger.
    Missing evidence is a review fallback, and all applicable reasons are retained.
    """
    reasons: list[str] = []
    if assessment.confidence_score < CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")

    amount = assessment.extracted_entities.disputed_amount
    if amount is not None and amount > DISPUTED_AMOUNT_THRESHOLD_USD:
        reasons.append("high_disputed_amount")

    if not evidence:
        reasons.append("missing_policy_evidence")
    elif max(item.similarity_score for item in evidence) < RETRIEVAL_SIMILARITY_THRESHOLD:
        reasons.append("low_policy_similarity")

    if assessment.legal_threat:
        reasons.append("legal_threat")
    if assessment.sentiment == "extreme_frustration":
        reasons.append("extreme_frustration")

    required = bool(reasons)
    return RoutingDecision(
        human_in_loop_required=required,
        escalation_reason=reasons,
        automation_eligible=not required,
    )


def route_failure(reason: str) -> RoutingDecision:
    """Fail closed when ingestion, retrieval, or model processing cannot complete."""
    if not reason.strip():
        reason = "processing_failure"
    return RoutingDecision(
        human_in_loop_required=True,
        escalation_reason=[reason],
        automation_eligible=False,
    )
