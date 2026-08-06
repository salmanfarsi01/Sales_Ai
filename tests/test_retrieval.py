import tempfile
import unittest
from pathlib import Path

from copilot.retrieval import LocalKnowledgeBase


class RetrievalTests(unittest.TestCase):
    def test_returns_relevant_source(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "resume.md").write_text(
                "Built real-time Python systems using WebSockets and asyncio.", encoding="utf-8"
            )
            Path(directory, "unrelated.txt").write_text("Landscape gardening notes.", encoding="utf-8")
            results = LocalKnowledgeBase(directory).search("Tell me about Python WebSocket experience")
            self.assertEqual(results[0].source, str(Path(directory, "resume.md").as_posix()))

    def test_empty_query_has_no_results(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(LocalKnowledgeBase(directory).search("a"), [])


if __name__ == "__main__":
    unittest.main()
