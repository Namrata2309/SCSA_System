"""SQLite persistence for idempotent processed tickets and compact audit output."""

import sqlite3
from contextlib import closing
from pathlib import Path

from .schemas import EnrichedTicket


class SQLiteTicketStore:
    """Store enriched output and audit metadata keyed by ticket ID.

    Raw ticket messages and customer IDs are not persisted. Reprocessing an
    existing ticket ID returns its original result, making retries idempotent.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS processed_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    human_in_loop_required INTEGER NOT NULL DEFAULT 1
                    )"""
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(processed_tickets)")
                }
                if "human_in_loop_required" not in columns:
                    connection.execute(
                        "ALTER TABLE processed_tickets ADD COLUMN human_in_loop_required INTEGER NOT NULL DEFAULT 1"
                    )
                    for ticket_id, result_json in connection.execute(
                        "SELECT ticket_id, result_json FROM processed_tickets"
                    ).fetchall():
                        result = EnrichedTicket.model_validate_json(result_json)
                        connection.execute(
                            "UPDATE processed_tickets SET human_in_loop_required = ? WHERE ticket_id = ?",
                            (int(result.human_in_loop_required), ticket_id),
                        )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_processed_review ON processed_tickets(human_in_loop_required, created_at DESC)"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def get(self, ticket_id: str) -> EnrichedTicket | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT result_json FROM processed_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return EnrichedTicket.model_validate_json(row[0]) if row else None

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        human_in_loop_required: bool | None = None,
    ) -> list[EnrichedTicket]:
        """Return newest processed records, optionally filtered to review status."""
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        where = ""
        parameters: list[int] = []
        if human_in_loop_required is not None:
            where = "WHERE human_in_loop_required = ?"
            parameters.append(int(human_in_loop_required))
        parameters.extend([limit, offset])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT result_json FROM processed_tickets {where}
                    ORDER BY created_at DESC, ticket_id LIMIT ? OFFSET ?""",
                parameters,
            ).fetchall()
        return [EnrichedTicket.model_validate_json(row[0]) for row in rows]

    def save(self, result: EnrichedTicket) -> EnrichedTicket:
        """Insert once, then return the canonical record (including on races)."""
        payload = result.model_dump_json()
        created_at = result.audit.created_at.isoformat() if result.audit else ""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """INSERT INTO processed_tickets
                       (ticket_id, result_json, created_at, human_in_loop_required)
                       VALUES (?, ?, ?, ?) ON CONFLICT(ticket_id) DO NOTHING""",
                    (
                        result.ticket_id,
                        payload,
                        created_at,
                        int(result.human_in_loop_required),
                    ),
                )
            row = connection.execute(
                "SELECT result_json FROM processed_tickets WHERE ticket_id = ?",
                (result.ticket_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Ticket result could not be read after saving")
        return EnrichedTicket.model_validate_json(row[0])
