"""Application assembly shared by CLI and local HTTP server."""

from pathlib import Path

from .llm import AssessmentProvider, UnavailableProvider, create_provider
from .pipeline import SupportPipeline
from .retrieval import load_retriever
from .storage import SQLiteTicketStore


def build_pipeline(
    *,
    policy_path: str | Path = "data/policies.jsonl",
    database_path: str | Path = "data/scsa.sqlite3",
    provider_name: str | None = None,
    env_file: str | Path | None = ".env",
    provider: AssessmentProvider | None = None,
) -> SupportPipeline:
    """Assemble local services; unavailable credentials always route to review."""
    if provider is None:
        try:
            provider = create_provider(provider_name, env_file=env_file)
        except (ValueError, OSError):
            provider = UnavailableProvider()
    retriever = load_retriever(policy_path)
    store = SQLiteTicketStore(database_path)
    return SupportPipeline(retriever, provider, store)
