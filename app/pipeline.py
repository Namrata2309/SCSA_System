"""End-to-end orchestration for SCSA ticket processing."""

import logging
from datetime import datetime, timezone
from typing import Protocol

from .governance import route_ticket
from .llm import AssessmentProvider, ModelProviderError
from .retrieval import TfidfRetriever
from .schemas import (
    AuditRecord,
    CustomerTicket,
    EnrichedTicket,
    ExtractedEntities,
    RoutingDecision,
)


_LOGGER = logging.getLogger(__name__)


class TicketStore(Protocol):
    def get(self, ticket_id: str) -> EnrichedTicket | None: ...
    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        human_in_loop_required: bool | None = None,
    ) -> list[EnrichedTicket]: ...
    def save(self, result: EnrichedTicket) -> EnrichedTicket: ...


class SupportPipeline:
    """Run retrieval, assessment, deterministic governance, and audit storage."""

    def __init__(
        self,
        retriever: TfidfRetriever,
        provider: AssessmentProvider,
        store: TicketStore | None = None,
    ):
        self.retriever = retriever
        self.provider = provider
        self.store = store

    def process(self, ticket: CustomerTicket) -> EnrichedTicket:
        if self.store:
            existing = self.store.get(ticket.ticket_id)
            if existing:
                return existing

        retrieval_error = False
        try:
            evidence = self.retriever.search(ticket.raw_message)
        except Exception:
            evidence = []
            retrieval_error = True

        try:
            if retrieval_error:
                raise ModelProviderError("policy_retrieval_failure")
            assessment = self.provider.assess(ticket, evidence)
            matching_evidence = [item for item in evidence if item.category == assessment.category]
            decision = route_ticket(assessment, matching_evidence)
            category = assessment.category
            sentiment = assessment.sentiment
            entities = assessment.extracted_entities
            confidence = assessment.confidence_score
            draft = assessment.auto_response_draft
        except Exception as error:
            _LOGGER.error(
                "Ticket assessment or routing failed; using human-review fallback "
                "(provider=%s, error_type=%s)",
                getattr(self.provider, "model_version", "unknown"),
                type(error).__name__,
            )
            reasons = ["model_provider_failure"]
            if retrieval_error:
                reasons = ["policy_retrieval_failure"]
            elif not evidence:
                reasons.append("missing_policy_evidence")
            decision = RoutingDecision(
                human_in_loop_required=True,
                escalation_reason=reasons,
                automation_eligible=False,
            )
            category = "unknown"
            sentiment = "neutral"
            entities = ExtractedEntities()
            confidence = 0.0
            draft = "A support specialist must review this ticket before a response can be drafted."

        now = datetime.now(timezone.utc)
        policy_ids = [item.policy_id for item in evidence]
        audit = AuditRecord(
            ticket_id=ticket.ticket_id,
            retrieved_policy_ids=policy_ids,
            retrieval_scores=[item.similarity_score for item in evidence],
            model_version=getattr(self.provider, "model_version", "unknown"),
            decision="human_review" if decision.human_in_loop_required else "automation_eligible",
            reasons=decision.escalation_reason,
            created_at=now,
        )
        result = EnrichedTicket(
            ticket_id=ticket.ticket_id,
            category=category,
            sentiment=sentiment,
            extracted_entities=entities,
            confidence_score=confidence,
            human_in_loop_required=decision.human_in_loop_required,
            escalation_reason=decision.escalation_reason,
            auto_response_draft=draft,
            retrieved_policy_ids=policy_ids,
            audit=audit,
        )
        return self.store.save(result) if self.store else result
