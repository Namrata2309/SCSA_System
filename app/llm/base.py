"""Provider-neutral assessment contract and safe failure handling."""

from dataclasses import dataclass
from typing import Protocol

from ..governance import route_failure
from ..schemas import CustomerTicket, ModelAssessment, PolicyEvidence, RoutingDecision


class ModelProviderError(RuntimeError):
    """Raised when a provider cannot produce a valid assessment."""


class UnavailableProvider:
    """Placeholder used when configuration is absent; always fails closed."""

    model_version = "unavailable"

    def __init__(self, reason: str = "provider_not_configured"):
        self.reason = reason

    def assess(
        self,
        ticket: CustomerTicket,
        evidence: list[PolicyEvidence],
    ) -> ModelAssessment:
        raise ModelProviderError(self.reason)


class AssessmentProvider(Protocol):
    """Common contract implemented by mock and live model providers."""

    model_version: str

    def assess(
        self,
        ticket: CustomerTicket,
        evidence: list[PolicyEvidence],
    ) -> ModelAssessment:
        """Classify a ticket using bounded policy context and return validated data."""


@dataclass(frozen=True)
class SafeAssessmentResult:
    """Assessment result or a review decision when model processing fails."""

    assessment: ModelAssessment | None
    failure_decision: RoutingDecision | None = None


def assess_or_review(
    provider: AssessmentProvider,
    ticket: CustomerTicket,
    evidence: list[PolicyEvidence],
) -> SafeAssessmentResult:
    """Turn provider failures into a deterministic human-review fallback."""
    try:
        return SafeAssessmentResult(assessment=provider.assess(ticket, evidence))
    except ModelProviderError:
        return SafeAssessmentResult(
            assessment=None,
            failure_decision=route_failure("model_provider_failure"),
        )
