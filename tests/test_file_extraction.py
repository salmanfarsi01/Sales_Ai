"""Tests for RAG file extraction system."""

import io
from pathlib import Path

import pytest

from copilot.file_extraction import ExtractedContent, FileExtractor


class TestPDFExtraction:
    """Test PDF extraction."""
    
    def test_extract_pdf_basic(self, tmp_path):
        """Test basic PDF extraction."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")
        
        # Create simple PDF
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Sample PDF content\nSecond line")
        doc.save(str(pdf_path))
        doc.close()
        
        # Extract
        result = FileExtractor.extract(str(pdf_path))
        
        assert isinstance(result, ExtractedContent)
        assert "Sample PDF content" in result.text
        assert result.file_type == "pdf"
        assert "file_name" in result.metadata


class TestTextExtraction:
    """Test text file extraction."""
    
    def test_extract_txt(self, tmp_path):
        """Test TXT extraction."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Hello\nWorld\nTest content", newline="\n")
        
        result = FileExtractor.extract(str(txt_path))
        
        assert result.text == "Hello\nWorld\nTest content"
        assert result.file_type == "txt"
    
    def test_extract_markdown(self, tmp_path):
        """Test Markdown extraction."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Title\n\nContent here", newline="\n")
        
        result = FileExtractor.extract(str(md_path))
        
        assert "Title" in result.text
        assert result.file_type == "md"


class TestCSVExtraction:
    """Test CSV extraction."""
    
    def test_extract_csv(self, tmp_path):
        """Test CSV extraction."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Name,Age,City\nJohn,30,NYC\nJane,25,LA", newline="\n")
        
        result = FileExtractor.extract(str(csv_path))
        
        assert "John" in result.text
        assert "30" in result.text
        assert result.file_type == "csv"


class TestUnsupportedFormat:
    """Test unsupported format handling."""
    
    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        unknown_path = tmp_path / "test.xyz"
        unknown_path.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            FileExtractor.extract(str(unknown_path))


class TestBlobExtraction:
    """Test extraction from binary blobs."""
    
    def test_extract_from_blob(self):
        """Test extracting from binary content."""
        content = b"This is text content"
        
        result = FileExtractor.extract("test.txt", file_content=content)
        
        assert "This is text content" in result.text
        assert result.file_type == "txt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
