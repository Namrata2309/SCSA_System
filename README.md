# Smart Customer Support Agent

Prototype for policy-grounded support triage, response drafting, deterministic escalation, and evaluation.

## What this project demonstrates

- Ingest and validate mock customer tickets.
- Retrieve relevant policy text from a local knowledge base (start with a transparent TF-IDF baseline).
- Use an LLM provider adapter to return validated structured classification, entities, confidence, and a response draft.
- Apply deterministic routing rules independently of the model.
- Evaluate schema integrity, intent classification, groundedness, and escalation behavior against labeled examples.

This is a local assessment prototype. It does not issue refunds, change accounts, or send customer messages.

## Requirements

- Python 3.11 or newer.
- Exact dependency versions are pinned in `requirements.txt` for reproducible setup.
- Optional LLM API credentials for live model mode. Offline/mock mode should remain available for reproducible demos.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

For live mode, set `LLM_PROVIDER=gemini`, `groq`, or `openai`; `LLM_MODEL` is optional. Gemini defaults to `gemini-3.8-flash` and reads `GEMINI_API_KEY`. Groq reads `GROQ_API_KEY` or `LLM_API_KEY`; OpenAI reads `OPENAI_API_KEY`. Never commit `.env` or real customer data. If credentials are absent or the provider fails, the pipeline routes the ticket to human review.

Gemini uses the official [`google-genai` SDK structured output](https://googleapis.github.io/python-genai/) and validates the result with `ModelAssessment`. The adapter leaves the SDK client deadline at its default unless an explicit timeout is supplied in code; transient provider failures receive one bounded retry, then fail closed to human review. The default `gemini-3.8-flash` model is listed as stable and supports structured outputs in [Google's model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash). Free-tier access and quotas depend on the account, region, and Google's current limits. Google states free-tier content may be used to improve products; use synthetic tickets only for Gemini free-tier demos ([Gemini API pricing and data use](https://ai.google.dev/gemini-api/docs/pricing)).

## Build status

Implementation is proceeding in the phases listed in `BUILD_PLAN.md`. **M0 (contracts and fixtures), M1 (retrieval baseline), M2 (structured model adapter), M3 (pipeline, persistence, CLI, and API), and M4 (evaluation and hardening) are implemented.** M5 project documentation is prepared; clean-environment reproduction and the candidate's 5–8 minute recording remain pending. The recording outline is provided separately and is not stored in this repository. Human claim adjudication is an optional labeled input to the groundedness metric; the evaluator reports it as unmeasured until that file is populated.

Install the pinned dependency with `pip install -r requirements.txt`. Current contracts, routing, and retrieval are in `app/schemas.py`, `app/governance.py`, and `app/retrieval.py`; fixture files are under `data/`.

## Run

The current implementation provides validated contracts, governance rules, policy retrieval, mock/live model adapters, and a synthetic evaluation CLI. Run its tests with:

```powershell
python -m unittest discover -v
```

Latest local run: 55 tests discovered, 54 passed, and the local HTTP route integration test could not connect because the environment denied loopback sockets (`WinError 10013`). This is not a full-suite pass. Run the suite in an environment that allows local sockets to verify that integration test.

Use the processing CLI with the offline mock provider:

```powershell
$dbPath = Join-Path $env:TEMP ("scsa-mock-" + [guid]::NewGuid().ToString("N") + ".sqlite3")
python -m app.cli process --input data/sample_tickets.jsonl --provider mock --db $dbPath
```

The unique temporary database ensures this run processes the fixtures from scratch instead of returning earlier idempotent results. The demo fixtures are synthetic. Use `--provider gemini` only when intentionally making a live API request; free-tier access and quotas vary, and only synthetic ticket data should be sent.

Start the local browser demo and API in mock mode:

```powershell
$dbPath = Join-Path $env:TEMP ("scsa-demo-" + [guid]::NewGuid().ToString("N") + ".sqlite3")
python -m app.cli serve --provider mock --db $dbPath
```

Open `http://127.0.0.1:8000`. The interface provides synthetic scenario buttons, ticket analysis, policy excerpts and similarity scores, a readable routing decision, and the human-review queue. The `serve` command also exposes `GET /health`, `GET /policies/search?q=...`, and the ticket API routes below. The interface shows whether the server is using mock or a live provider; when using Gemini, Groq, or OpenAI, ticket text is sent to that provider. Use synthetic data for demos. No mode sends a reply to the customer.

Run the reproducible offline evaluation with `python -m app.cli evaluate --provider mock --pricing data/model_pricing.json`. It writes a JSON report to stdout; add `--output report.json` to save it. The checked-in 10-ticket mock baseline in `reports/evaluation-mock-2026-10-30.json` measures schema validity 1.00, intent accuracy 0.80 (macro-F1 0.738), and escalation precision 0.556 / recall 1.00 / F1 0.714 with zero false negatives. These are results on a small synthetic set, not production performance. The report also includes latency and retries; mock mode has no observed model tokens or observed usage cost. Scenario costs are planning estimates based on the assumptions in `data/model_pricing.json`, not measured usage. Groundedness remains `not_adjudicated` until human claim reviews are added to `data/groundedness_reviews.jsonl`; the report includes each draft, evidence, and current SHA-256 hash to support review. Each review row needs `ticket_id`, `draft_sha256`, `reviewer`, and `claims` with `claim` and a disposition of `supported`, `contradicted`, or `unsupported`. Stale reviews are rejected. For a live provider run, supply that provider's current rates with `--input-price-per-million` and `--output-price-per-million` or a matching pricing card.

### Ticket API routes

- `GET /health`: liveness response.
- `GET /policies/search?q=...`: top three evidence excerpts and similarity scores for the local demo UI.
- `POST /tickets`: validate and process one ticket; repeated `ticket_id` values return the existing idempotent result.
- `GET /tickets/{ticket_id}`: retrieve a processed result and audit metadata.
- `GET /tickets`: page through processed results with `limit` (1–200) and `offset`.
- `GET /tickets?status=human_review` or `?status=automation_eligible`: filter by routing status. `human_in_loop_required=true|false` is also supported.
- `GET /review-queue`: list tickets requiring human review; accepts the same pagination parameters.

Ticket submission uses the `CustomerTicket` fields described below. Invalid payloads return a review-required validation response; unknown ticket IDs return 404. The API prepares drafts and review records only; it does not send replies or perform account actions. In the external CLI and API enriched-ticket response, `escalation_reason` is a readable string (with all triggered reasons separated by semicolons; routine tickets say `No escalation required.`). Internal pipeline results, persisted records, and the nested audit record retain the individual reason codes as lists.

## Architecture

```text
Ticket JSON/HTTP
      |
      v
Input validation -> Policy retrieval -> LLM provider adapter -> Output schema validation
                                                   |
                                                   v
                           Deterministic governance and routing
                                                   |
                         +-------------------------+------------------+
                         |                                            |
                   Draft + eligible                         Human review queue
                         +-------------------------+------------------+
                                                   v
                                   Audit record + evaluation
```

Suggested modules: `schemas`, `ingestion`, `retrieval`, `llm`, `governance`, `pipeline`, `api`/`cli`, `evaluation`, and `storage`. Keep provider-specific code behind an interface. Begin with TF-IDF and local JSON/SQLite; introduce a vector database or queue only when measured needs justify them. See `ARCHITECTURE.md`.

## Mandatory escalation rules

Escalate if any condition is true, preserving every applicable reason:

1. Confidence score `< 0.75`.
2. Disputed amount `> $100`.
3. Best policy retrieval similarity `< 0.60`.
4. Legal threat or extreme customer frustration is detected.

At exactly `0.75`, exactly `$100`, and exactly `0.60`, that respective threshold does not trigger. Missing amounts remain unknown. Invalid model output, unavailable model, or missing evidence must fail safely into human review; these are additional safeguards, not replacements for the four required rules.

## Data contracts

- `support_tickets`: `ticket_id`, `customer_id`, `raw_message`, `timestamp`.
- `policy_kb`: `policy_id`, `category`, `policy_text`.
- Enriched external response: `ticket_id`, `category`, `sentiment`, `extracted_entities` (`order_id`, `disputed_amount`, and optionally dates), `confidence_score` in `[0,1]`, `human_in_loop_required`, string `escalation_reason`, and `auto_response_draft`. The audit object keeps reason codes as a list.

Use Pydantic models for input and output validation. Keep retrieved policy IDs/text and rule outcomes in an audit record, even if the externally returned JSON follows the assignment's required shape.

## Evaluation

Run the automated evaluation script over a fixed, labeled sample set. Report schema validity, intent macro-F1 and per-class metrics, hallucination/unsupported-claim rate with an explicit evidence rubric, and escalation precision/recall plus false-negative count. Include boundary tests and simultaneous-trigger tests. Do not claim a target score without measured results. See `EVALUATION_PLAN.md`.

## Cost and latency

The Groq GPT-OSS 20B price-card snapshot in `data/model_pricing.json` was checked on 2026-10-30 against the [official model pricing page](https://console.groq.com/docs/model/openai/gpt-oss-20b). At $0.075/M input and $0.30/M output tokens, the illustrative model-only estimates are $0.15 / $0.30 / $0.60 per 1,000 tickets for low/base/high assumed token volumes (1,000/250, 2,000/500, and 4,000/1,000 input/output tokens per ticket). These are estimates, not observed token counts. They assume zero retries and exclude hosting, database, network, tax, and labor. Replace the snapshot with current rates for a different model. The report separately records measured local latency and provider usage when available; mock model latency is not representative of live inference.

## Security and limitations

- Treat ticket content as untrusted input and defend against prompt injection.
- Keep API keys in environment variables; redact sensitive values from logs.
- Model confidence is a signal, not calibrated probability unless calibration is measured.
- A draft is not evidence that an action was taken. Do not send it automatically in this prototype.
- Mock data and a small evaluation set do not establish production readiness.

## Submission checklist

- [ ] Modular Python repository, environment example, and dependency file.
- [ ] Mock ticket and policy datasets.
- [ ] Pydantic contracts and safe failure behavior.
- [ ] Local policy retrieval with a defined score.
- [ ] Four deterministic escalation triggers and tests.
- [ ] Evaluation script and measured report.
- [ ] Setup, architecture, cost/latency assumptions documented.
- [ ] 5–8 minute recorded demo showing both an automation-eligible grounded case and human-review cases (recording outline provided separately; no video or runbook is included here).
- [ ] Prepare to discuss scaling from 100 to 100,000 tickets/day.

## Supporting documents

- `BUILD_PLAN.md`: sequence, milestones, and exit criteria.
- `ARCHITECTURE.md`: component boundaries, data flow, and scaling path.
- `EVALUATION_PLAN.md`: datasets, metrics, and acceptance checks.
