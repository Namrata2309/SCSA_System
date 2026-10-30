"""Reproducible evaluation for the labeled synthetic ticket set."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .governance import route_ticket
from .retrieval import load_retriever
from .schemas import CustomerTicket, ModelAssessment


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _classification_metrics(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = sorted(set(expected) | set(predicted))
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for actual, guess in zip(expected, predicted):
        matrix[actual][guess] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][guess] for guess in labels if guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": sum(matrix[label].values()),
        }
    size = len(expected)
    return {
        "accuracy": sum(a == p for a, p in zip(expected, predicted)) / size if size else 0.0,
        "macro_f1": statistics.mean(item["f1"] for item in per_class.values()) if per_class else 0.0,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(a and p for a, p in zip(expected, predicted))
    tn = sum(not a and not p for a, p in zip(expected, predicted))
    fp = sum(not a and p for a, p in zip(expected, predicted))
    fn = sum(a and not p for a, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "false_negatives": fn,
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
    }


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def _scenario_costs(pricing: dict[str, Any] | None) -> dict[str, Any] | None:
    if pricing is None:
        return None
    input_rate = float(pricing["input_usd_per_million_tokens"])
    output_rate = float(pricing["output_usd_per_million_tokens"])
    scenarios = {}
    for name, usage in pricing["scenario_token_assumptions_per_ticket"].items():
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        per_ticket = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        scenarios[name] = {
            "assumed_input_tokens_per_ticket": input_tokens,
            "assumed_output_tokens_per_ticket": output_tokens,
            "estimated_cost_per_ticket_usd": per_ticket,
            "estimated_cost_per_1000_tickets_usd": per_ticket * 1000,
        }
    return {
        "provider": pricing["provider"],
        "model": pricing["model"],
        "pricing_checked_on": pricing["checked_on"],
        "source_url": pricing["source_url"],
        "input_usd_per_million_tokens": input_rate,
        "output_usd_per_million_tokens": output_rate,
        "scenarios": scenarios,
        "assumptions": pricing["assumptions"],
    }


def _groundedness(reviews_path: Path, draft_hashes: dict[str, str]) -> dict[str, Any]:
    reviewed: list[dict[str, Any]] = []
    if reviews_path.exists():
        for line_no, line in enumerate(reviews_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                tid = row["ticket_id"]
                if tid not in draft_hashes or row.get("draft_sha256") != draft_hashes[tid]:
                    raise ValueError("draft hash missing or does not match current output")
                if not row.get("reviewer"):
                    raise ValueError("reviewer is required")
                for claim in row["claims"]:
                    if claim.get("disposition") not in {"supported", "contradicted", "unsupported"} or not claim.get("claim"):
                        raise ValueError("each claim needs text and an allowed disposition")
                reviewed.append(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid groundedness review on line {line_no}: {error}") from error
    claims = [claim for row in reviewed for claim in row["claims"]]
    factual_count = len(claims)
    unsupported = sum(claim["disposition"] == "unsupported" for claim in claims)
    contradicted = sum(claim["disposition"] == "contradicted" for claim in claims)
    return {
        "status": "reviewed" if factual_count else "not_adjudicated",
        "reviewed_tickets": len(reviewed),
        "factual_claims_reviewed": factual_count,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "unsupported_rate": unsupported / factual_count if factual_count else None,
        "rubric": "Human reviewers label each factual claim supported, contradicted, or unsupported against retrieved policy text and ticket facts. No automated judge is used.",
    }


def evaluate(
    *,
    dataset_path: str | Path,
    policy_path: str | Path,
    provider: Any,
    groundedness_path: str | Path | None = None,
    model_input_price_per_million: float | None = None,
    model_output_price_per_million: float | None = None,
    pricing_path: str | Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run one deterministic pass and return a JSON-serializable report."""
    dataset = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    pricing = json.loads(Path(pricing_path).read_text(encoding="utf-8")) if pricing_path else None
    if pricing is not None:
        model_input_price_per_million = float(pricing["input_usd_per_million_tokens"])
        model_output_price_per_million = float(pricing["output_usd_per_million_tokens"])
    retriever = load_retriever(policy_path)
    expected_intents: list[str] = []
    predicted_intents: list[str] = []
    expected_review: list[bool] = []
    predicted_review: list[bool] = []
    retrieval_times: list[float] = []
    model_times: list[float] = []
    end_to_end_times: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    retries = 0
    errors = 0
    valid = 0
    invalid = 0
    fallbacks = 0
    drafts: dict[str, str] = {}
    draft_hashes: dict[str, str] = {}
    class_distribution: Counter[str] = Counter()
    review_inputs: list[dict[str, Any]] = []

    for row in dataset:
        ticket = CustomerTicket.model_validate(row["ticket"])
        expected_intents.append(row["expected_category"])
        class_distribution[row["expected_category"]] += 1
        expected_review.append(bool(row["expected_human_review"]))
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        evidence = retriever.search(ticket.raw_message)
        retrieval_times.append(time.perf_counter() - retrieval_start)
        model_start = time.perf_counter()
        assessment: ModelAssessment | None = None
        try:
            raw_assessment = provider.assess(ticket, evidence)
            assessment = ModelAssessment.model_validate(raw_assessment.model_dump() if isinstance(raw_assessment, ModelAssessment) else raw_assessment)
            valid += 1
        except Exception:
            invalid += 1
            fallbacks += 1
            errors += 1
        model_times.append(time.perf_counter() - model_start)
        retries += int(getattr(provider, "last_retry_count", 0))
        usage = getattr(provider, "last_usage", None)
        if isinstance(usage, dict):
            if isinstance(usage.get("input_tokens"), int):
                input_tokens.append(usage["input_tokens"])
            if isinstance(usage.get("output_tokens"), int):
                output_tokens.append(usage["output_tokens"])
        if assessment:
            predicted_intents.append(assessment.category)
            matching_evidence = [item for item in evidence if item.category == assessment.category]
            decision = route_ticket(assessment, matching_evidence)
            predicted_review.append(decision.human_in_loop_required)
            drafts[ticket.ticket_id] = assessment.auto_response_draft
            draft_hashes[ticket.ticket_id] = hashlib.sha256(assessment.auto_response_draft.encode("utf-8")).hexdigest()
        else:
            predicted_intents.append("unknown")
            predicted_review.append(True)
            drafts[ticket.ticket_id] = ""
            draft_hashes[ticket.ticket_id] = hashlib.sha256(b"").hexdigest()
        review_inputs.append({
            "ticket_id": ticket.ticket_id,
            "ticket_text": ticket.raw_message,
            "draft": drafts[ticket.ticket_id],
            "draft_sha256": draft_hashes[ticket.ticket_id],
            "retrieved_evidence": [item.model_dump(mode="json") for item in evidence],
        })
        end_to_end_times.append(time.perf_counter() - start)

    n = len(dataset)
    tokens_in = sum(input_tokens) if input_tokens else None
    tokens_out = sum(output_tokens) if output_tokens else None
    cost = None
    if tokens_in is not None and tokens_out is not None and model_input_price_per_million is not None and model_output_price_per_million is not None:
        cost = tokens_in * model_input_price_per_million / 1_000_000 + tokens_out * model_output_price_per_million / 1_000_000
    report = {
        "report_version": "1",
        "evaluated_on": date.today().isoformat(),
        "dataset": str(dataset_path),
        "dataset_version": "synthetic-v1",
        "provider": getattr(provider, "model_version", type(provider).__name__),
        "command": command,
        "sample_count": n,
        "class_distribution": dict(sorted(class_distribution.items())),
        "schema_validity": {"valid_outputs": valid, "total_outputs": n, "validity_fraction": valid / n if n else None, "invalid_output_count": invalid, "safe_fallback_count": fallbacks},
        "intent": _classification_metrics(expected_intents, predicted_intents),
        "escalation": _binary_metrics(expected_review, predicted_review),
        "groundedness": _groundedness(Path(groundedness_path) if groundedness_path else Path("data/groundedness_reviews.jsonl"), draft_hashes),
        "groundedness_review_inputs": review_inputs,
        "operations": {
            "latency_seconds": {
                "end_to_end_median": statistics.median(end_to_end_times) if end_to_end_times else None,
                "end_to_end_p95": _percentile95(end_to_end_times),
                "retrieval_median": statistics.median(retrieval_times) if retrieval_times else None,
                "retrieval_p95": _percentile95(retrieval_times),
                "model_median": statistics.median(model_times) if model_times else None,
                "model_p95": _percentile95(model_times),
            },
            "input_tokens_observed": tokens_in,
            "output_tokens_observed": tokens_out,
            "retry_count": retries,
            "retry_rate_per_ticket": retries / n if n else None,
            "provider_error_count": errors,
            "estimated_cost_usd": cost,
            "cost_per_1000_tickets_usd": cost * 1000 / n if cost is not None and n else None,
            "pricing": {"input_usd_per_million_tokens": model_input_price_per_million, "output_usd_per_million_tokens": model_output_price_per_million},
            "pricing_scenario_estimates": _scenario_costs(pricing),
        },
        "limitations": ["Synthetic set is small and is a development baseline, not production evidence.", "Mock-provider evaluation has no observed model token usage; model latency measures local mock execution only.", "Groundedness is reported only for human-adjudicated claims whose draft hash matches this run."],
    }
    return report
