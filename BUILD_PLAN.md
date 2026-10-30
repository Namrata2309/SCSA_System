# Build Plan

## Objective

Deliver a reproducible Python prototype that ingests mock support tickets, retrieves policy evidence, generates validated structured assessments and response drafts, routes risky cases to human review using deterministic rules, and reports measured evaluation results.

## Current implementation status

- **M0 complete:** Pydantic contracts, timezone-aware ticket timestamps, deterministic escalation rules and safe failure routing, 19 synthetic tickets, six policy records, and a separate labeled evaluation fixture are implemented. Tests cover schema constraints and rule boundaries/multiple triggers.
- **M1 complete:** a standard-library TF-IDF retriever returns inspectable policy IDs, categories, excerpts, and corpus-specific cosine scores. Empty corpora and zero-overlap queries return no evidence.
- **Validation:** `python -m unittest discover -v` passes 14 tests on Python 3.14.5.
- **M2 complete:** provider-neutral adapter contract, deterministic offline mock, OpenAI and Groq Responses structured-output adapters, schema validation, bounded transient retries, and fail-closed review fallback are implemented. Unit tests use fake HTTP transports; a live API call has not been made.
- **Validation:** `python -m unittest discover -v` passes 23 tests on Python 3.14.5.
- **M3 complete:** JSON/JSONL ingestion, retrieval-model-governance pipeline, idempotent SQLite results/audit, JSONL CLI, health and ticket submission, ticket lookup/listing, and human review queue HTTP endpoints are implemented. Groq is supported to match the local provider setting; missing credentials fail closed to review.
- **Validation:** `python -m unittest discover -v` passes 33 tests on Python 3.14.5, including local HTTP route integration; tests use synthetic fixtures and fake provider transports.
- **M4 implementation:** a self-contained labeled evaluation set, offline evaluation CLI, schema/intent/escalation metrics, groundedness review intake bound to draft hashes, observed latency/retry/token reporting, and published-price low/base/high model-cost scenarios. The standard demo fixture now includes a routine policy-grounded ticket that qualifies for automation. Live-model evaluation remains opt-in.
- **M4 baseline:** `python -m app.cli evaluate --provider mock --pricing data/model_pricing.json --output reports/evaluation-mock-2026-10-30.json` reports schema validity 1.00, intent accuracy 0.80 (macro-F1 0.738), escalation precision 0.556 / recall 1.00 / F1 0.714, and zero false negatives. Groq GPT-OSS 20B model-only costs are estimated at $0.15 / $0.30 / $0.60 per 1,000 tickets under documented low/base/high token assumptions; actual model usage cost is unmeasured in mock mode. Groundedness remains unadjudicated until human claim labels are supplied.
- **Validation:** `python -m unittest discover -v` passes 44 tests on Python 3.14.5.
- **Assignment alignment pass (2026-10-30):** mock drafts only quote same-category evidence; absent same-category evidence uses a conservative draft and routes to review. Order ID extraction accepts numeric and alphanumeric order formats and ignores ordinary prose. CLI/API output uses a readable string for `escalation_reason`, while internal results, persistence, and audit retain reason-code lists. The mock baseline remains schema validity 1.00, intent accuracy 0.80 (macro-F1 0.738), escalation precision 0.556 / recall 1.00 / F1 0.714, zero false negatives, and groundedness `not_adjudicated`. The refreshed report is `reports/evaluation-mock-2026-10-30.json`.
- **Latest validation:** `python -m unittest discover -v` passes 49 tests on Python 3.14.5, including the local HTTP integration test. Offline mock processing of all 19 demo tickets produced a policy-grounded automation-eligible routine case (`T-1019`); review cases remained in review. No live provider request was made.
- **Gemini provider:** Added an injectable Google Gen AI SDK adapter using `gemini-3.8-flash` by default, JSON structured output, Pydantic validation, bounded retries, fail-closed routing, and optional SDK usage metadata. Provider selection uses `GEMINI_API_KEY` and `LLM_MODEL`; mocked unit tests cover configuration, valid and invalid outputs, provider failures, and human-review fallback. Free-tier availability, quotas, and data-use terms are documented; demos should use synthetic tickets only.
- **Live Gemini smoke check (2026-10-30):** A user-run synthetic shipping question returned a schema-valid draft and classification. It correctly routed to human review because the retrieved-policy similarity was 0.2908, below the required 0.60 threshold. This is one smoke example, not an evaluation result; no live provider call was made during the latest local validation.
- **M5 project preparation:** Added a browser demo served by the existing local server, with sample ticket scenarios, category-labeled policy evidence, routing explanations, and a human-review queue. README commands and dependency pinning were aligned with the implementation; a unique temporary SQLite path is documented for fresh mock runs; recording instructions are delivered separately. Recording and clean-environment reproduction remain outstanding; M5 is not complete until the candidate records the assignment's 5–8 minute walkthrough and independently verifies setup.
- **Latest local validation after M5 doc/dependency changes (Python 3.14.5):** Full discovery found 55 tests: 54 passed, and the local HTTP integration test was blocked by sandbox Windows socket permissions (`WinError 10013`); this is not a full-suite pass. A fresh-database mock CLI run processed all 19 tickets (one automation-eligible, 18 human review). Mock evaluation processed 10 tickets with schema validity 1.00, intent accuracy 0.80, escalation recall 1.00, and groundedness `not_adjudicated`. No live provider call was made during these checks.

## Scope and priority

### Required for assignment

1. Ticket ingestion and parsing.
2. Local policy knowledge base and retrieval-augmented context.
3. Structured LLM classification and response drafting.
4. Pydantic/JSON output validation.
5. Human review decision based on all four deterministic triggers.
6. Automated evaluation of schema, hallucination/groundedness, or intent metrics (implement a useful selection of all three).
7. Mock data, setup instructions, and a 5–8 minute recorded walkthrough.

### Engineering enhancements

Audit records, idempotent ticket processing, API endpoints beyond the demo path, SQLite persistence, bounded retries, and production scaling notes.

## Proposed repository layout

```text
smart-customer-support-agent/
  app/
    static/
      index.html
    schemas.py
    ingestion.py
    retrieval.py
    llm/
      base.py
      mock.py
      provider.py
    governance.py
    pipeline.py
    api.py
    cli.py
    storage.py
  data/
    policies.jsonl
    sample_tickets.jsonl
    evaluation.jsonl
  tests/
    test_schemas.py
    test_governance.py
    test_pipeline.py
    test_evaluation.py
  ARCHITECTURE.md
  EVALUATION_PLAN.md
  .env.example
  .gitignore
  requirements.txt
  README.md
```

## Milestones

### M0 — Contracts and fixtures (Days 1–2)

- Create repository, Python environment, dependency file, and configuration example.
- Define ticket, policy, extracted entities, model assessment, routing result, and audit schemas.
- Build 15–20 diverse mock tickets and 5–10 concise policy records, plus a separately labeled evaluation set.
- Implement and test all four escalation rules including exact-boundary and multiple-trigger cases.

**Exit:** Inputs/outputs validate; all rule tests pass; fixtures are deterministic and contain no real personal data.

### M1 — Retrieval baseline (Day 3)

- Normalize policy text and build a TF-IDF retriever.
- Return policy ID, category, excerpt, and a documented similarity score.
- Define behavior for empty corpus and no relevant match.

**Exit:** Relevant policy ranking is inspectable; similarity score and threshold semantics are documented.

### M2 — Structured model adapter (Days 4–5)

- Define provider-neutral adapter; create mock adapter for offline repeatability.
- Prompt the model with retrieved evidence and untrusted-ticket boundaries.
- Parse and validate structured response; reject malformed or out-of-range output.
- Add timeout and bounded retry behavior for transient failures; fail closed to review.

**Exit:** Mock mode works without credentials; live mode returns schema-valid output or a safe review result.

### M3 — End-to-end pipeline and audit (Days 6–7)

- Connect ingestion, retrieval, model, governance, routing, and persistence.
- Aggregate all triggered reasons; maintain evidence references and version metadata.
- Provide CLI processing and a minimal health/ticket API if time allows.

**Exit:** One low-risk ticket and each escalation scenario traverse the full pipeline; no unsafe automatic route on errors.

### M4 — Evaluation and hardening (Days 8–9)

- Implement schema validity, intent metrics, groundedness review, and escalation metrics.
- Run boundary, missing-value, injection-like, no-policy, malformed-output, and provider-failure cases.
- Record latency and token usage; calculate cost estimate per 1,000 tickets from current model pricing.

**Implementation:** `python -m app.cli evaluate --provider mock --pricing data/model_pricing.json` emits a JSON report with cost scenarios based on an explicitly dated published price card and assumed token volumes. Human groundedness reviews are stored in `data/groundedness_reviews.jsonl`; unsupported claims are not inferred from a model judge. Observed token cost is separate and remains null in mock mode. Live-model token measurements and human-adjudicated groundedness remain optional evidence to improve the baseline.

**Exit:** Reproducible report includes sample size, method, measured results, and limitations.

### M5 — Submission package (Day 10)

- Finalize and cross-check README, architecture, evaluation, and dependency setup against the implementation.
- Verify a fresh mock CLI run and evaluation; reproduce setup in a clean environment.
- Candidate records a 5–8 minute walkthrough showing an automation-eligible grounded case and human-review cases. The video and its recording instructions are kept outside this repository.
- Prepare scaling and tradeoff discussion.

**Status:** Project-side documentation and setup refinements are prepared. Clean-environment reproduction and the candidate's recording are still pending, so M5 is not yet complete.

**Exit:** Repository and recorded demo follow the candidate deliverables checklist in the assignment.

## Definition of done

- Assignment schemas and mandatory behaviors are implemented.
- All four rules use strict comparisons exactly as specified: `<0.75`, `>$100`, `<0.60`, and legal threat/extreme frustration.
- Every applicable reason is recorded.
- Invalid or unavailable model output routes to review.
- Automated evaluation is reproducible and claims only observed results.
- README setup commands match the actual project.
- Demo shows both an eligible routine case and several review cases.
