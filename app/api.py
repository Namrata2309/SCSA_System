"""Local demo page and HTTP API for ticket processing and policy inspection."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Type
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import ValidationError

from .pipeline import SupportPipeline
from .serialization import escalation_reason_text, external_ticket_payload
from .schemas import CustomerTicket


MAX_REQUEST_BYTES = 1_000_000
MAX_PAGE_SIZE = 200


def process_ticket_payload(
    pipeline: SupportPipeline,
    payload: object,
) -> tuple[int, dict]:
    """Validate and process a single JSON payload without exposing internals."""
    try:
        ticket = CustomerTicket.model_validate(payload)
    except (ValidationError, TypeError):
        return 422, {
            "human_in_loop_required": True,
            "escalation_reason": escalation_reason_text(["invalid_ticket"]),
        }
    try:
        return 200, external_ticket_payload(pipeline.process(ticket))
    except Exception:
        return 503, {
            "ticket_id": ticket.ticket_id,
            "human_in_loop_required": True,
            "escalation_reason": escalation_reason_text(["processing_failure"]),
        }


def fetch_ticket_payload(
    pipeline: SupportPipeline,
    ticket_id: str,
) -> tuple[int, dict]:
    """Fetch one processed ticket result by its ID."""
    store = pipeline.store
    if store is None:
        return 503, {"error": "storage_unavailable"}
    result = store.get(ticket_id)
    if result is None:
        return 404, {"error": "ticket_not_found"}
    return 200, external_ticket_payload(result)


def list_ticket_payloads(
    pipeline: SupportPipeline,
    query: dict[str, list[str]],
    *,
    review_only: bool = False,
) -> tuple[int, dict]:
    """List processed tickets with bounded pagination and route filtering."""
    store = pipeline.store
    if store is None or not hasattr(store, "list"):
        return 503, {"error": "storage_unavailable"}
    try:
        limit = int(query.get("limit", ["100"])[0])
        offset = int(query.get("offset", ["0"])[0])
    except (TypeError, ValueError):
        return 400, {"error": "invalid_pagination"}
    if not 1 <= limit <= MAX_PAGE_SIZE or offset < 0:
        return 400, {"error": "invalid_pagination", "max_limit": MAX_PAGE_SIZE}

    status = query.get("status", [None])[0]
    review_flag = query.get("human_in_loop_required", [None])[0]
    if status not in (None, "human_review", "automation_eligible"):
        return 400, {"error": "invalid_status"}
    if review_flag not in (None, "true", "false"):
        return 400, {"error": "invalid_review_filter"}
    if status is not None and review_flag is not None:
        if (status == "human_review") != (review_flag == "true"):
            return 400, {"error": "conflicting_review_filter"}
    if review_only:
        if status == "automation_eligible" or review_flag == "false":
            return 400, {"error": "conflicting_review_filter"}
        review = True
    elif status is not None:
        review = status == "human_review"
    elif review_flag is not None:
        review = review_flag == "true"
    else:
        review = None

    records = store.list(
        limit=limit,
        offset=offset,
        human_in_loop_required=review,
    )
    return 200, {
        "tickets": [external_ticket_payload(record) for record in records],
        "count": len(records),
        "limit": limit,
        "offset": offset,
    }


def search_policy_payload(pipeline: SupportPipeline, query: str) -> tuple[int, dict]:
    """Return a small evidence preview for the local demo interface."""
    query = query.strip()
    if not query or len(query) > 2_000:
        return 400, {"error": "invalid_policy_query"}
    try:
        evidence = pipeline.retriever.search(query, top_k=3)
    except Exception:
        return 503, {"error": "policy_retrieval_failure"}
    return 200, {
        "query": query,
        "policies": [item.model_dump(mode="json") for item in evidence],
    }


def make_handler(pipeline: SupportPipeline) -> Type[BaseHTTPRequestHandler]:
    """Create a request handler bound to a configured support pipeline."""
    class TicketHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_demo_page(self) -> None:
            page_path = Path(__file__).parent / "static" / "index.html"
            try:
                content = page_path.read_bytes()
            except OSError:
                self._send(503, {"error": "demo_interface_unavailable"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send_demo_page()
                return
            if parsed.path == "/health":
                model_version = getattr(pipeline.provider, "model_version", "unknown")
                provider_mode = (
                    "mock" if str(model_version).startswith("mock")
                    else "unavailable" if model_version == "unavailable"
                    else "live"
                )
                self._send(200, {
                    "status": "ok",
                    "provider_mode": provider_mode,
                    "model_version": model_version,
                })
                return
            if parsed.path == "/policies/search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                status, response = search_policy_payload(pipeline, query)
                self._send(status, response)
                return
            if parsed.path == "/tickets":
                status, response = list_ticket_payloads(
                    pipeline,
                    parse_qs(parsed.query),
                )
                self._send(status, response)
                return
            if parsed.path == "/review-queue":
                status, response = list_ticket_payloads(
                    pipeline,
                    parse_qs(parsed.query),
                    review_only=True,
                )
                self._send(status, response)
                return
            if parsed.path.startswith("/tickets/"):
                ticket_id = unquote(parsed.path[len("/tickets/"):])
                if not ticket_id or "/" in ticket_id:
                    self._send(404, {"error": "ticket_not_found"})
                    return
                status, response = fetch_ticket_payload(pipeline, ticket_id)
                self._send(status, response)
                return
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/tickets":
                self._send(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = MAX_REQUEST_BYTES + 1
            if size <= 0 or size > MAX_REQUEST_BYTES:
                self._send(413 if size > MAX_REQUEST_BYTES else 400, {
                    "human_in_loop_required": True,
                    "escalation_reason": escalation_reason_text(["invalid_request_size"]),
                })
                return
            try:
                payload = json.loads(self.rfile.read(size))
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                self._send(422, {
                    "human_in_loop_required": True,
                    "escalation_reason": escalation_reason_text(["invalid_ticket"]),
                })
                return
            status, response = process_ticket_payload(pipeline, payload)
            self._send(status, response)

        def log_message(self, format: str, *args) -> None:
            # Ticket messages and identifiers should not be written to access logs.
            return

    return TicketHandler


def make_server(
    pipeline: SupportPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Build the synchronous local demo server."""
    return ThreadingHTTPServer((host, port), make_handler(pipeline))
