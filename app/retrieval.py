"""Local TF-IDF policy retrieval with a documented cosine similarity score."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schemas import PolicyEvidence, PolicyRecord


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> list[str]:
    """Lowercase text and tokenize contiguous ASCII letters and digits."""
    return _TOKEN_RE.findall(text.lower())


def load_policies(path: str | Path) -> list[PolicyRecord]:
    """Load JSONL policy records, rejecting malformed lines with line context."""
    records: list[PolicyRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(PolicyRecord.model_validate_json(line))
            except Exception as error:
                raise ValueError(f"Invalid policy record on line {line_number}: {error}") from error
    return records


class TfidfRetriever:
    """Rank policy records by L2-normalized TF-IDF cosine similarity.

    Document term frequency is raw count. Smoothed inverse document frequency is
    ``log((1 + document_count) / (1 + document_frequency)) + 1``. Query and
    document vectors use the same corpus IDF and are L2-normalized; the reported
    cosine scores therefore fall in [0, 1]. The score is corpus-specific and is
    not a probability. A query with no shared terms returns an empty list.
    """

    def __init__(self, policies: Iterable[PolicyRecord]):
        self.policies = list(policies)
        self._tokens = [normalize_text(policy.policy_text) for policy in self.policies]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self.policies)
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._vector(tokens) for tokens in self._tokens]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        frequencies = Counter(tokens)
        weighted = {
            term: frequency * self._idf[term]
            for term, frequency in frequencies.items()
            if term in self._idf
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in weighted.items()}

    def search(self, query: str, *, top_k: int = 3) -> list[PolicyEvidence]:
        """Return highest scoring evidence, ordered by score then policy ID.

        Empty corpora, empty queries, and queries with no corpus term in common
        return no evidence. Nonzero scores below the governance threshold remain
        visible so the caller can inspect evidence and route the ticket safely.
        """
        if top_k <= 0 or not self.policies:
            return []
        query_vector = self._vector(normalize_text(query))
        if not query_vector:
            return []

        ranked: list[tuple[float, PolicyRecord]] = []
        for policy, vector in zip(self.policies, self._vectors):
            score = sum(
                weight * vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )
            if score > 0:
                ranked.append((score, policy))
        ranked.sort(key=lambda item: (-item[0], item[1].policy_id))
        return [
            PolicyEvidence(
                policy_id=policy.policy_id,
                category=policy.category,
                excerpt=policy.policy_text[:240],
                similarity_score=min(score, 1.0),
            )
            for score, policy in ranked[:top_k]
        ]


def load_retriever(path: str | Path) -> TfidfRetriever:
    """Create a retriever from a JSONL policy file."""
    return TfidfRetriever(load_policies(path))
