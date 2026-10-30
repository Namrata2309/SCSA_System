import unittest
from pathlib import Path

from app.retrieval import TfidfRetriever, load_policies
from app.schemas import PolicyRecord


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.policies = load_policies(ROOT / "data" / "policies.jsonl")
        self.retriever = TfidfRetriever(self.policies)

    def test_relevant_policy_ranks_first(self):
        results = self.retriever.search("When will my approved refund reach my card?")
        self.assertTrue(results)
        self.assertEqual(results[0].policy_id, "POL-REF-001")
        self.assertGreater(results[0].similarity_score, 0)

    def test_scores_are_bounded_and_ordered(self):
        results = self.retriever.search("return unused item delivered")
        scores = [item.similarity_score for item in results]
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_overlap_returns_no_evidence(self):
        self.assertEqual(self.retriever.search("volcanic telescope"), [])

    def test_empty_corpus_and_query_are_safe(self):
        self.assertEqual(TfidfRetriever([]).search("refund"), [])
        self.assertEqual(self.retriever.search("!!!"), [])

    def test_top_k_and_nonpositive_top_k(self):
        self.assertLessEqual(len(self.retriever.search("order delivery tracking refund", top_k=2)), 2)
        self.assertEqual(self.retriever.search("refund", top_k=0), [])

    def test_same_score_tie_breaks_by_policy_id(self):
        records = [
            PolicyRecord(policy_id="B", category="x", policy_text="refund card"),
            PolicyRecord(policy_id="A", category="x", policy_text="refund card"),
        ]
        results = TfidfRetriever(records).search("refund card")
        self.assertEqual([item.policy_id for item in results], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
