# Architecture Notes

## Design

Use a modular monolith: one Python application with separately testable components. This is enough for the assessment and avoids introducing deployment complexity before throughput or availability needs are measured.

## Data flow

1. **Ingest:** Parse ticket JSON/HTTP payload and validate required fields and timestamp.
2. **Retrieve:** Search local policy records. Return top evidence with policy IDs and similarity scores.
3. **Classify and draft:** Pass ticket plus bounded policy context to a provider adapter. Request category, sentiment, entities, confidence, and a customer-safe draft as structured data.
4. **Validate:** Validate model response using Pydantic. Treat malformed output and model errors as review-required.
5. **Govern:** Run pure deterministic rules outside the prompt. Collect all applicable trigger reasons.
6. **Route:** Mark eligible routine tickets as automation-eligible; all triggered or failed cases go to a human review state. The prototype only prepares drafts; it does not send them.
7. **Audit/evaluate:** Store input reference, retrieved policy IDs/scores, schema/model version, decision, reasons, timestamps, and evaluation labels with sensitive data minimized.

## Component boundaries

| Component | Responsibility | Key contract |
|---|---|---|
| Ingestion | Parse and validate ticket | `CustomerTicket` |
| Knowledge base | Load policy records | `PolicyRecord` |
| Retriever | Rank relevant records | Evidence list with documented score |
| Model adapter | Provider-neutral structured inference | `ModelAssessment` |
| Governance | Apply fixed business rules | `RoutingDecision` with all reasons |
| Pipeline | Orchestrate stages and safe failures | `EnrichedTicket` plus audit metadata |
| Demo interface | Serve a same-origin browser client for local ticket review | Static page calling the local API |
| Evaluation | Compare output to labels/evidence | Metrics report |
| Storage | Local JSON/SQLite records | Idempotent ticket key |

## Technical choices

- **Retrieval:** Begin with TF-IDF for transparency and no external vector service. The baseline lowercases and tokenizes ASCII letters/digits, uses raw term counts and smoothed IDF `log((1 + N) / (1 + df)) + 1`, then L2-normalizes document and query vectors and reports their cosine similarity. Scores are corpus-specific, not probabilities. Zero-overlap queries return no evidence; nonzero scores below `0.60` remain visible and trigger governance review. Compare an embedding retriever only if evaluation shows a measurable retrieval gap.
- **Model:** `app/llm/` defines a provider-neutral adapter, a deterministic offline mock, OpenAI/Groq Responses adapters, and a Google Gemini adapter using the official `google-genai` SDK with JSON structured output. Ticket content is serialized as untrusted data and the model has no tools. Validate provider output again with `ModelAssessment`; refusal, malformed output, missing credentials, and exhausted transient retries map to human review. Confidence remains a routing signal, not a calibrated probability. Gemini token usage is recorded when present in SDK response metadata and otherwise remains unavailable.
- **Validation:** Pydantic models enforce required fields, ranges, and types. Reject or safely route invalid responses.
- **Governance:** Pure Python rules; prompt text cannot override them.
- **Persistence:** JSON fixtures plus local SQLite for idempotent enriched results and audit/review state. Raw ticket messages and customer IDs are not stored. Production would use a managed relational database.
- **Execution:** Synchronous request/CLI path initially. The local HTTP server also serves a lightweight browser demo; the page submits tickets and displays retrieved policy evidence and review decisions. Move to a durable queue when measured load or latency requires asynchronous work.

## Assignment routing rules

- Confidence `< 0.75` => review.
- Disputed amount `> 100 USD` => review.
- Retrieval similarity `< 0.60` => review.
- Legal threat or extreme frustration => review.

Reasons are cumulative. At equality, a numeric threshold does not trigger. Missing amount or missing retrieval evidence is not interpreted as zero/high confidence; apply a safe review fallback.

## Scaling discussion: 100 to 100,000 tickets/day

At 100/day, a synchronous modular application and local database are adequate for a prototype. At higher volume, place ingestion behind an API gateway and durable queue, use stateless workers, add retry/dead-letter handling and idempotency, store policies and audit data in managed services, and scale retrieval/model calls independently. Add rate limits, concurrency controls, backpressure, PII-safe observability, model-provider fallback policy, and alerting. Load test and set service objectives before selecting infrastructure. Do not claim a specific cloud design or cost without workload and pricing assumptions.

## Risks and controls

- **Unsupported policy claims:** cite retrieved policy IDs in internal trace; evaluate draft claims against evidence.
- **Prompt injection:** mark ticket text as untrusted data; do not grant model tools or let it change rules.
- **Missed escalation:** measure escalation recall and inspect false negatives.
- **Bad confidence calibration:** treat confidence as a routing signal and document that calibration is unproven.
- **Provider outage:** timeout, bounded retry, safe human-review fallback.
- **Sensitive data leakage:** synthetic fixtures, secret management, log redaction, minimal retention.
- **Gemini free tier:** access and quotas can depend on the account, region, and current Google limits. Google may use free-tier content to improve its products, so use synthetic tickets only for free-tier demos.
