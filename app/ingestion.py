"""JSON and JSONL ticket ingestion with line-specific validation errors."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import CustomerTicket


def parse_ticket(payload: dict[str, Any]) -> CustomerTicket:
    """Validate a parsed JSON object as a customer ticket."""
    return CustomerTicket.model_validate(payload)


def load_tickets(path: str | Path) -> list[CustomerTicket]:
    """Load tickets from a JSON array/object or newline-delimited JSON file."""
    source_path = Path(path)
    # Accept UTF-8 files written by Windows PowerShell 5.1, which prepends a BOM.
    raw = source_path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return []

    if source_path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {source_path}: {error.msg}") from error
        items = payload if isinstance(payload, list) else [payload]
        tickets: list[CustomerTicket] = []
        for index, item in enumerate(items, start=1):
            try:
                tickets.append(parse_ticket(item))
            except (ValidationError, TypeError) as error:
                raise ValueError(f"Invalid ticket {index} in {source_path}: {error}") from error
        return tickets

    tickets = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            tickets.append(parse_ticket(payload))
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            raise ValueError(f"Invalid ticket on line {line_number} in {source_path}: {error}") from error
    return tickets
