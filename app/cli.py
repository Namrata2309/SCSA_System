"""Command-line entry points for ticket processing and local API service."""

import argparse
import json
import sys
from pathlib import Path

from .application import build_pipeline
from .ingestion import load_tickets
from .evaluation import evaluate
from .llm import create_provider
from .serialization import external_ticket_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scsa")
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", help="process tickets from a JSON or JSONL file")
    process.add_argument("--input", required=True, help="path to ticket JSON/JSONL")
    process.add_argument("--policies", default="data/policies.jsonl")
    process.add_argument("--db", default="data/scsa.sqlite3", help="SQLite audit/result store")
    process.add_argument("--provider", choices=("mock", "openai", "groq", "gemini"))

    serve = commands.add_parser("serve", help="start the local API and browser demo interface")
    serve.add_argument("--policies", default="data/policies.jsonl")
    serve.add_argument("--db", default="data/scsa.sqlite3")
    serve.add_argument("--provider", choices=("mock", "openai", "groq", "gemini"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    evaluation = commands.add_parser("evaluate", help="evaluate a provider against the labeled synthetic set")
    evaluation.add_argument("--input", default="data/evaluation.jsonl")
    evaluation.add_argument("--policies", default="data/policies.jsonl")
    evaluation.add_argument("--provider", choices=("mock", "openai", "groq", "gemini"), default="mock")
    evaluation.add_argument("--reviews", default="data/groundedness_reviews.jsonl")
    evaluation.add_argument("--pricing", help="published rate card and low/base/high token assumptions")
    evaluation.add_argument("--output", help="optional path for the JSON report")
    evaluation.add_argument("--input-price-per-million", type=float)
    evaluation.add_argument("--output-price-per-million", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        provider = create_provider(args.provider)
        pricing_path = args.pricing
        if pricing_path is None and (
            args.provider == "mock" or getattr(provider, "model_version", None) == "openai/gpt-oss-20b"
        ):
            pricing_path = "data/model_pricing.json"
        report = evaluate(
            dataset_path=args.input,
            policy_path=args.policies,
            provider=provider,
            groundedness_path=args.reviews,
            pricing_path=pricing_path,
            model_input_price_per_million=args.input_price_per_million,
            model_output_price_per_million=args.output_price_per_million,
            command="python -m app.cli " + " ".join(sys.argv[1:] if argv is None else argv),
        )
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    pipeline = build_pipeline(
        policy_path=Path(args.policies),
        database_path=Path(args.db),
        provider_name=args.provider,
    )
    if args.command == "serve":
        from .api import make_server

        server = make_server(pipeline, args.host, args.port)
        try:
            print(f"SCSA API listening at http://{args.host}:{args.port}", file=sys.stderr)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    tickets = load_tickets(args.input)
    for ticket in tickets:
        result = pipeline.process(ticket)
        print(json.dumps(external_ticket_payload(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
