"""Deterministic offline provider for demos, tests, and development."""

import re

from ..schemas import (
    CustomerTicket,
    ExtractedEntities,
    ModelAssessment,
    PolicyEvidence,
)


_CATEGORY_TERMS = {
    "damaged_item": ("damaged", "cracked", "broken"),
    "refunds": ("refund", "reimburse", "money back"),
    "returns": ("return", "send back", "final-sale", "final sale"),
    "billing": ("charged", "charge", "authorization", "fee", "payment", "billing"),
    "shipping": ("shipping", "delivery", "delayed", "late", "package", "tracking"),
    "order_status": ("order", "where is", "status", "tracking link"),
}
_LEGAL_RE = re.compile(r"\b(sue|lawsuit|lawyer|attorney|legal action)\b", re.IGNORECASE)
_EXTREME_RE = re.compile(
    r"\b(furious|outraged|unacceptable|ridiculous|third time|fed up|angry)\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"(?:\$\s*|\bUSD\s*)(\d+(?:,\d{3})*(?:\.\d{1,2})?)", re.IGNORECASE)
_ORDER_RE = re.compile(
    r"\border\s*(?:(?:number|no\.?)\s*)?#?\s*((?:[a-z]{2,}-?\d{3,}|\d{3,}))\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class MockProvider:
    """Rule-based, reproducible provider that never makes external requests."""

    model_version = "mock-rules-v1"

    def __init__(self):
        self.last_usage = None
        self.last_retry_count = 0

    def assess(
        self,
        ticket: CustomerTicket,
        evidence: list[PolicyEvidence],
    ) -> ModelAssessment:
        message = ticket.raw_message
        lowered = message.lower()
        category = next(
            (
                name
                for name, terms in _CATEGORY_TERMS.items()
                if any(term in lowered for term in terms)
            ),
            "other",
        )
        legal_threat = bool(_LEGAL_RE.search(message))
        extreme_frustration = bool(_EXTREME_RE.search(message))
        if extreme_frustration:
            sentiment = "extreme_frustration"
        elif any(term in lowered for term in ("thank", "great", "perfect", "appreciate")):
            sentiment = "positive"
        elif any(term in lowered for term in ("late", "charged", "damaged", "broken", "wrong", "not arrived")):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        amount_match = _AMOUNT_RE.search(message)
        disputed_amount = (
            float(amount_match.group(1).replace(",", ""))
            if amount_match
            else None
        )
        order_match = _ORDER_RE.search(message)
        entities = ExtractedEntities(
            order_id=order_match.group(1) if order_match else None,
            disputed_amount=disputed_amount,
            dates=_DATE_RE.findall(message),
        )
        matching_evidence = next((item for item in evidence if item.category == category), None)
        if matching_evidence:
            draft = (
                "Thanks for contacting us. Our current guidance says: "
                f"{matching_evidence.excerpt} Please reply with any details that would help us understand your case."
            )
        else:
            draft = (
                "Thanks for contacting us. I do not have a matching policy excerpt to answer this confidently. "
                "Please share any details that may help our support team review your question."
            )

        confidence = 0.88 if category != "other" else 0.55
        if matching_evidence is None:
            confidence = min(confidence, 0.70)
        return ModelAssessment(
            category=category,
            sentiment=sentiment,
            extracted_entities=entities,
            confidence_score=confidence,
            legal_threat=legal_threat,
            auto_response_draft=draft,
        )
