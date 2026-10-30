# Evaluation Plan

## Purpose

Measure whether the prototype follows its data contract, classifies labeled examples, grounds drafts in policy evidence, and routes risky tickets to human review. A small mock set is a development signal, not proof of production performance.

## Dataset protocol

- Keep evaluation labels separate from prompt examples where practical.
- Include a representative mix of intents, sentiment levels, entity presence/absence, ambiguous questions, and escalation categories.
- Record expected intent, entities, risk labels, expected route, and supporting policy IDs.
- Report sample count and class distribution. Use synthetic data only.
- Version the dataset and scoring rubric so results can be reproduced.

## Metrics

### Schema validity

`valid_outputs / total_outputs`. Also report invalid-output count and safe-fallback count. Count malformed responses as invalid even when downstream fallback is safe.

### Intent classification

Build a confusion matrix and report accuracy, per-class precision/recall/F1, and macro-F1. Macro-F1 prevents common categories from hiding poor performance on rare categories.

### Escalation behavior

Compare predicted review route with expected risk labels. Report precision, recall, F1, confusion matrix, and false-negative count. Prioritize recall for high-risk cases; review every false negative in a small dataset.

### Groundedness / hallucination proxy

Define an unsupported claim as a factual policy or action claim in the draft that is not supported by retrieved policy evidence or the ticket. Annotate each draft claim as supported, contradicted, or unsupported using a written rubric. Report unsupported claims divided by factual claims reviewed, along with denominator and sample size. Human adjudication is appropriate for a small prototype; do not label an LLM judge as ground truth.

### Operations

Record end-to-end latency (median and p95), retrieval latency, model latency, input/output token counts, retry rate, and provider errors. Derive cost per 1,000 tickets using the selected provider's current published rates and observed token counts; state all assumptions.

## Required test cases

- Confidence just below, equal to, and above 0.75.
- Amount below, exactly equal to, and above $100.
- Similarity below, exactly equal to, and above 0.60.
- Legal threat and extreme frustration.
- Several simultaneous triggers; retain every reason.
- Missing disputed amount and empty/no-match policy corpus.
- Invalid schema, provider timeout, and malformed JSON.
- Customer text containing instruction-like prompt injection.
- Routine grounded case eligible for automation review policy.

## Acceptance and reporting

Acceptance is behavioral: schemas are enforced, threshold comparisons match the assignment, multiple reasons are preserved, and failure paths require human review. Establish quality baselines from measured data; do not invent accuracy goals or present sample metrics without running the evaluation. Include known limitations, dataset size, date, and exact command used in the final report.

## Current implementation and run

The labeled synthetic set is `data/evaluation.jsonl`. Run the offline baseline with its model cost estimate using:

```powershell
python -m app.cli evaluate --provider mock --pricing data/model_pricing.json --output reports/evaluation-mock-YYYY-MM-DD.json
```

The report contains each output draft, its SHA-256 digest, ticket text, and retrieved evidence under `groundedness_review_inputs`. Reviewers can copy those records into `data/groundedness_reviews.jsonl`; each JSONL row has `ticket_id`, `draft_sha256`, `reviewer`, and a `claims` array (`claim`, `disposition`). Dispositions are `supported`, `contradicted`, and `unsupported`. Reports reject stale hashes. Until a human reviewer supplies these labels, the groundedness rate is null and status is `not_adjudicated`.

For a live provider run, select `--provider gemini`, `--provider groq`, or `--provider openai`. Gemini usage is included when the SDK returns token metadata and remains unavailable otherwise. Measured usage cost is calculated only when usage is returned and both `--input-price-per-million` and `--output-price-per-million` are supplied. Scenario estimates from a price card remain separate from observed usage. Mock timings represent local execution, not model latency, and mock mode reports no observed token usage. Gemini free-tier availability and quotas vary by account, region, and current Google limits; because free-tier content may be used to improve Google products, run only synthetic tickets in free-tier demos.
