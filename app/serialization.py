"""External response formatting while keeping internal reason codes intact."""

from .schemas import EnrichedTicket


_REASON_LABELS = {
    "low_confidence": "Confidence is below 0.75",
    "high_disputed_amount": "Disputed amount exceeds $100",
    "missing_policy_evidence": "No matching policy evidence is available",
    "low_policy_similarity": "Policy similarity is below 0.60",
    "legal_threat": "Legal threat detected",
    "extreme_frustration": "Extreme customer frustration detected",
    "model_provider_failure": "Model or provider failure",
    "policy_retrieval_failure": "Policy retrieval failure",
    "invalid_ticket": "Ticket validation failed",
    "processing_failure": "Ticket processing failed",
    "invalid_request_size": "Request size is invalid",
}


def escalation_reason_text(reasons: list[str]) -> str:
    """Render every internal reason code as one readable external string."""
    if not reasons:
        return "No escalation required."
    return "; ".join(
        _REASON_LABELS.get(reason, reason.replace("_", " ").capitalize())
        for reason in reasons
    )


def external_ticket_payload(result: EnrichedTicket) -> dict:
    """Serialize an enriched ticket using the assignment's string reason field."""
    payload = result.model_dump(mode="json")
    payload["escalation_reason"] = escalation_reason_text(result.escalation_reason)
    return payload
