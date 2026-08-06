import unittest

from copilot.knowledge_upload import safe_stem


class PdfKnowledgeTests(unittest.TestCase):
    def test_sanitizes_uploaded_filename(self):
        self.assertEqual(safe_stem("../../My Resume (Final).pdf"), "My_Resume_Final")

    def test_supplies_fallback_name(self):
        self.assertEqual(safe_stem("....pdf"), "document")


if __name__ == "__main__":
    unittest.main()
