import unittest

from copilot.knowledge_upload import is_pdf_signature, safe_stem


class PdfKnowledgeTests(unittest.TestCase):
    def test_sanitizes_uploaded_filename(self):
        self.assertEqual(safe_stem("../../My Resume (Final).pdf"), "My_Resume_Final")

    def test_supplies_fallback_name(self):
        self.assertEqual(safe_stem("....pdf"), "document")

    def test_accepts_pdf_with_leading_whitespace_before_header(self):
        data = bytes([
            0, 13, 10, 9, 37, 80, 68, 70, 45, 49, 46, 52, 10, 37, 0, 255, 255,
        ])
        self.assertTrue(is_pdf_signature(data))

    def test_rejects_non_pdf_content(self):
        self.assertFalse(is_pdf_signature(b"hello world"))


if __name__ == "__main__":
    unittest.main()
