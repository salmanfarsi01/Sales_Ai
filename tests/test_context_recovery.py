import unittest

from copilot.context_recovery import needs_context, recover_query


class ContextRecoveryTests(unittest.TestCase):
    def test_merges_lowercase_continuation(self):
        self.assertEqual(
            recover_query("in detail?", ["Can you explain how the AI system works"]),
            "Can you explain how the AI system works in detail?",
        )

    def test_resolves_referential_question(self):
        self.assertEqual(
            recover_query("How does it work?", ["I wanted to know about Django."]),
            "I wanted to know about Django. How does it work?",
        )

    def test_keeps_standalone_question(self):
        question = "What authentication methods does the product support?"
        self.assertEqual(recover_query(question, ["Unrelated pricing discussion."]), question)

    def test_limits_history(self):
        recovered = recover_query("And that?", ["one topic", "two topic", "three topic"], max_turns=1)
        self.assertEqual(recovered, "three topic And that?")

    def test_detects_ambiguous_reference(self):
        self.assertTrue(needs_context("How does it work?"))


if __name__ == "__main__":
    unittest.main()
