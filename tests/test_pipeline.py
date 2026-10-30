import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from urllib.parse import quote
from urllib.request import Request, urlopen
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.api import (
    fetch_ticket_payload,
    list_ticket_payloads,
    make_server,
    process_ticket_payload,
)
from app.cli import main
from app.ingestion import load_tickets
from app.llm import MockProvider, ModelProviderError, UnavailableProvider
from app.pipeline import SupportPipeline
from app.retrieval import load_retriever
from app.schemas import CustomerTicket, ModelAssessment, PolicyEvidence
from app.serialization import external_ticket_payload
from app.storage import SQLiteTicketStore


ROOT = Path(__file__).resolve().parents[1]


def ticket(ticket_id="PIPE-1", message="Standard shipping estimates are 3 to 5 business days."):
    return CustomerTicket(
        ticket_id=ticket_id,
        customer_id="CUSTOMER-SYNTHETIC",
        raw_message=message,
        timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )


class CountingProvider:
    model_version = "counting-test-v1"

    def __init__(self):
        self.calls = 0

    def assess(self, ticket, evidence):
        self.calls += 1
        return MockProvider().assess(ticket, evidence)


class BrokenProvider:
    model_version = "broken-test-v1"

    def assess(self, ticket, evidence):
        raise ModelProviderError("simulated provider outage")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.retriever = load_retriever(ROOT / "data" / "policies.jsonl")

    def test_ingestion_loads_mock_fixture_and_rejects_bad_timestamp(self):
        tickets = load_tickets(ROOT / "data" / "sample_tickets.jsonl")
        self.assertEqual(len(tickets), 19)
        with tempfile.TemporaryDirectory() as folder:
            bom_file = Path(folder) / "bom.jsonl"
            bom_file.write_text(
                '{"ticket_id":"bom","customer_id":"c","raw_message":"Help",'
                '"timestamp":"2026-01-01T00:00:00Z"}\n',
                encoding="utf-8-sig",
            )
            self.assertEqual(load_tickets(bom_file)[0].ticket_id, "bom")

            bad_file = Path(folder) / "bad.jsonl"
            bad_file.write_text(
                '{"ticket_id":"x","customer_id":"y","raw_message":"z","timestamp":"2026-01-01"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_tickets(bad_file)

    def test_pipeline_routes_routine_ticket_and_creates_audit(self):
        pipeline = SupportPipeline(self.retriever, MockProvider())
        result = pipeline.process(ticket())
        self.assertFalse(result.human_in_loop_required)
        self.assertEqual(result.escalation_reason, [])
        self.assertEqual(result.audit.decision, "automation_eligible")
        self.assertIn("POL-SHP-001", result.audit.retrieved_policy_ids)
        self.assertEqual(result.audit.model_version, "mock-rules-v1")
        self.assertEqual(result.audit.retrieved_policy_ids, result.retrieved_policy_ids)

    def test_sample_fixture_includes_a_routine_automation_eligible_case(self):
        tickets = load_tickets(ROOT / "data" / "sample_tickets.jsonl")
        result = SupportPipeline(self.retriever, MockProvider()).process(tickets[-1])
        self.assertFalse(result.human_in_loop_required)
        self.assertEqual(result.escalation_reason, [])

    def test_pipeline_preserves_all_routing_triggers(self):
        risky = ticket(
            "PIPE-RISK",
            "I am furious and will sue over the disputed $125 refund for order 9812.",
        )
        result = SupportPipeline(self.retriever, MockProvider()).process(risky)
        self.assertTrue(result.human_in_loop_required)
        self.assertIn("high_disputed_amount", result.escalation_reason)
        self.assertIn("legal_threat", result.escalation_reason)
        self.assertIn("extreme_frustration", result.escalation_reason)
        self.assertEqual(result.audit.reasons, result.escalation_reason)

    def test_category_mismatch_fails_closed_and_keeps_wrong_policy_out_of_draft(self):
        class FixedRetriever:
            def search(self, _message):
                return [PolicyEvidence(
                    policy_id="POL-SHP-WRONG", category="shipping",
                    excerpt="Shipping policy says refunds are always immediate.",
                    similarity_score=0.95,
                )]

        result = SupportPipeline(FixedRetriever(), MockProvider()).process(
            ticket("PIPE-MISMATCH", "Where is my refund?"),
        )
        self.assertTrue(result.human_in_loop_required)
        self.assertIn("missing_policy_evidence", result.escalation_reason)
        self.assertIn("low_confidence", result.escalation_reason)
        self.assertNotIn("always immediate", result.auto_response_draft)

    def test_external_output_uses_string_while_audit_keeps_multiple_reason_codes(self):
        result = SupportPipeline(self.retriever, MockProvider()).process(
            ticket("PIPE-EXTERNAL", "I am furious and will sue over a $125 refund."),
        )
        payload = external_ticket_payload(result)
        self.assertIsInstance(payload["escalation_reason"], str)
        self.assertIn("Disputed amount exceeds $100", payload["escalation_reason"])
        self.assertIn("Legal threat detected", payload["escalation_reason"])
        self.assertIn("Extreme customer frustration detected", payload["escalation_reason"])
        self.assertIsInstance(result.escalation_reason, list)
        self.assertEqual(result.audit.reasons, result.escalation_reason)

    def test_unavailable_provider_and_no_evidence_fail_closed(self):
        result = SupportPipeline(self.retriever, UnavailableProvider()).process(
            ticket("PIPE-FAIL", "quasar"),
        )
        self.assertTrue(result.human_in_loop_required)
        self.assertEqual(
            result.escalation_reason,
            ["model_provider_failure", "missing_policy_evidence"],
        )
        self.assertEqual(result.category, "unknown")
        self.assertEqual(result.confidence_score, 0)

    def test_store_is_idempotent_and_does_not_save_raw_ticket(self):
        marker = "SYNTHETIC_PRIVATE_RAW_MESSAGE_42"
        private_ticket = ticket("PIPE-CACHE", f"{marker}: standard shipping time?")
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "audit.sqlite3"
            store = SQLiteTicketStore(database)
            provider = CountingProvider()
            pipeline = SupportPipeline(self.retriever, provider, store)
            first = pipeline.process(private_ticket)
            second = SupportPipeline(self.retriever, BrokenProvider(), store).process(private_ticket)
            self.assertEqual(first, second)
            self.assertEqual(provider.calls, 1)
            with closing(sqlite3.connect(database)) as connection:
                (saved_json,) = connection.execute(
                    "SELECT result_json FROM processed_tickets WHERE ticket_id = ?",
                    (private_ticket.ticket_id,),
                ).fetchone()
            self.assertNotIn(marker, saved_json)
            self.assertNotIn("CUSTOMER-SYNTHETIC", saved_json)

    def test_api_payload_validation_and_processing(self):
        pipeline = SupportPipeline(self.retriever, MockProvider())
        status, invalid = process_ticket_payload(pipeline, {"ticket_id": "bad"})
        self.assertEqual(status, 422)
        self.assertEqual(invalid["escalation_reason"], "Ticket validation failed")

        status, result = process_ticket_payload(pipeline, ticket().model_dump(mode="json"))
        self.assertEqual(status, 200)
        self.assertFalse(result["human_in_loop_required"])
        self.assertIsInstance(result["escalation_reason"], str)
        self.assertIn("audit", result)

        risky_ticket = ticket("API-MULTI", "I am furious and will sue over a $125 refund.")
        status, risky = process_ticket_payload(pipeline, risky_ticket.model_dump(mode="json"))
        self.assertEqual(status, 200)
        self.assertIsInstance(risky["escalation_reason"], str)
        self.assertIn("Disputed amount exceeds $100", risky["escalation_reason"])
        self.assertIn("Legal threat detected", risky["escalation_reason"])
        self.assertIsInstance(risky["audit"]["reasons"], list)
        self.assertGreaterEqual(len(risky["audit"]["reasons"]), 3)

    def test_api_detail_listing_and_review_queue_helpers(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteTicketStore(Path(folder) / "api.sqlite3")
            pipeline = SupportPipeline(self.retriever, MockProvider(), store)
            routine = pipeline.process(ticket("API-ROUTINE"))
            review = pipeline.process(ticket("API-REVIEW", "quasar"))

            status, found = fetch_ticket_payload(pipeline, routine.ticket_id)
            self.assertEqual(status, 200)
            self.assertEqual(found["ticket_id"], routine.ticket_id)
            self.assertIsInstance(found["escalation_reason"], str)
            self.assertEqual(fetch_ticket_payload(pipeline, "missing")[0], 404)

            status, all_results = list_ticket_payloads(pipeline, {})
            self.assertEqual(status, 200)
            self.assertEqual(all_results["count"], 2)

            status, review_results = list_ticket_payloads(
                pipeline, {"status": ["human_review"]},
            )
            self.assertEqual(status, 200)
            self.assertEqual([item["ticket_id"] for item in review_results["tickets"]], [review.ticket_id])
            self.assertEqual(list_ticket_payloads(pipeline, {"limit": ["999"]})[0], 400)
            self.assertEqual(
                list_ticket_payloads(
                    pipeline,
                    {"status": ["human_review"], "human_in_loop_required": ["false"]},
                )[0],
                400,
            )
            self.assertEqual(
                list_ticket_payloads(pipeline, {"status": ["automation_eligible"]}, review_only=True)[0],
                400,
            )

    def test_http_routes_health_submit_detail_list_and_review_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteTicketStore(Path(folder) / "http.sqlite3")
            pipeline = SupportPipeline(self.retriever, MockProvider(), store)
            server = make_server(pipeline, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            root = f"http://127.0.0.1:{server.server_port}"

            def get(path):
                with urlopen(root + path, timeout=2) as response:
                    return response.status, json.loads(response.read())

            def post(path, payload):
                request = Request(
                    root + path,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    return response.status, json.loads(response.read())

            try:
                status, health = get("/health")
                self.assertEqual((status, health["status"]), (200, "ok"))
                good_ticket = ticket("HTTP-ROUTINE")
                status, submitted = post("/tickets", good_ticket.model_dump(mode="json"))
                self.assertEqual(status, 200)
                self.assertFalse(submitted["human_in_loop_required"])

                status, detail = get(f"/tickets/{quote(good_ticket.ticket_id)}")
                self.assertEqual(status, 200)
                self.assertEqual(detail["ticket_id"], good_ticket.ticket_id)
                status, listed = get("/tickets?human_in_loop_required=false&limit=10")
                self.assertEqual((status, listed["count"]), (200, 1))

                risky_ticket = ticket("HTTP-REVIEW", "quasar")
                post("/tickets", risky_ticket.model_dump(mode="json"))
                status, queue = get("/review-queue")
                self.assertEqual((status, queue["count"]), (200, 1))
                self.assertEqual(queue["tickets"][0]["ticket_id"], risky_ticket.ticket_id)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


    def test_cli_process_outputs_json_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main([
                    "process",
                    "--input", str(ROOT / "data" / "sample_tickets.jsonl"),
                    "--db", str(Path(folder) / "cli.sqlite3"),
                    "--provider", "mock",
                ])
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 19)
        self.assertTrue(all("audit" in row for row in rows))
        self.assertTrue(all(isinstance(row["escalation_reason"], str) for row in rows))
        multi_reason = next(row for row in rows if row["ticket_id"] == "T-1015")
        self.assertIn("Legal threat detected", multi_reason["escalation_reason"])
        self.assertIn("Extreme customer frustration detected", multi_reason["escalation_reason"])
        self.assertIsInstance(multi_reason["audit"]["reasons"], list)
        self.assertFalse(rows[-1]["human_in_loop_required"])


if __name__ == "__main__":
    unittest.main()
