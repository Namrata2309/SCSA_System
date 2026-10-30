"""Validated data contracts for the SCSA pipeline."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerTicket(Contract):
    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    raw_message: str = Field(min_length=1)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


class PolicyRecord(Contract):
    policy_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    policy_text: str = Field(min_length=1)


class ExtractedEntities(Contract):
    order_id: str | None = None
    disputed_amount: float | None = Field(default=None, ge=0)
    dates: list[str] = Field(default_factory=list)


class ModelAssessment(Contract):
    category: str = Field(min_length=1)
    sentiment: Literal["positive", "neutral", "negative", "extreme_frustration"]
    extracted_entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    confidence_score: float = Field(ge=0, le=1)
    legal_threat: bool = False
    auto_response_draft: str = ""


class PolicyEvidence(Contract):
    policy_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    similarity_score: float = Field(ge=0, le=1)


class RoutingDecision(Contract):
    human_in_loop_required: bool
    escalation_reason: list[str] = Field(default_factory=list)
    automation_eligible: bool


class AuditRecord(Contract):
    ticket_id: str = Field(min_length=1)
    retrieved_policy_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    model_version: str | None = None
    schema_version: str = "1"
    decision: Literal["automation_eligible", "human_review"]
    reasons: list[str] = Field(default_factory=list)
    created_at: datetime


class EnrichedTicket(Contract):
    ticket_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    sentiment: str = Field(min_length=1)
    extracted_entities: ExtractedEntities
    confidence_score: float = Field(ge=0, le=1)
    human_in_loop_required: bool
    escalation_reason: list[str] = Field(default_factory=list)
    auto_response_draft: str
    retrieved_policy_ids: list[str] = Field(default_factory=list)
    audit: AuditRecord | None = None
