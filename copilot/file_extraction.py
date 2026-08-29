"""Multi-format file extraction utilities for RAG system."""

from __future__ import annotations

import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF for PDFs


@dataclass(frozen=True)
class ExtractedContent:
    """Result of file extraction."""
    text: str
    metadata: dict[str, str]
    file_type: str


class FileExtractor:
    """Extracts text from multiple file formats efficiently."""

    SUPPORTED_FORMATS = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }

    @staticmethod
    def extract(file_path: str | Path, file_content: Optional[bytes] = None) -> ExtractedContent:
        """Extract text from any supported file format.
        
        Args:
            file_path: Path to file (used for format detection)
            file_content: Binary file content. If None, reads from file_path
            
        Returns:
            ExtractedContent with extracted text and metadata
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if file_content is None:
            file_content = file_path.read_bytes()
        
        if suffix == ".pdf":
            return FileExtractor._extract_pdf(file_path, file_content)
        elif suffix in {".txt", ".md"}:
            return FileExtractor._extract_text(file_path, file_content)
        elif suffix == ".docx":
            return FileExtractor._extract_docx(file_path, file_content)
        elif suffix == ".doc":
            return FileExtractor._extract_doc(file_path, file_content)
        elif suffix == ".csv":
            return FileExtractor._extract_csv(file_path, file_content)
        elif suffix in {".xlsx", ".xls"}:
            return FileExtractor._extract_excel(file_path, file_content)
        elif suffix in {".jpg", ".jpeg", ".png"}:
            return FileExtractor._extract_image(file_path, file_content)
        elif suffix in {".mp4", ".webm"}:
            return FileExtractor._extract_video(file_path, file_content)
        elif suffix in {".wav", ".mp3"}:
            return FileExtractor._extract_audio(file_path, file_content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _extract_pdf(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract text from PDF using PyMuPDF."""
        text_parts = []
        metadata_dict = {}
        
        try:
            pdf_document = fitz.open(stream=content, filetype="pdf")
            metadata_dict = pdf_document.metadata or {}
            
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            
            pdf_document.close()
        except Exception as e:
            raise ValueError(f"Failed to extract PDF: {e}")
        
        return ExtractedContent(
            text="\n\n".join(text_parts),
            metadata={
                "file_name": file_path.name,
                "file_type": "pdf",
                "title": metadata_dict.get("title", ""),
                "author": metadata_dict.get("author", ""),
                "pages": str(len(text_parts)),
            },
            file_type="pdf"
        )

    @staticmethod
    def _extract_text(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract text from .txt and .md files."""
        text = content.decode("utf-8", errors="ignore")
        return ExtractedContent(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
            },
            file_type=file_path.suffix.lower().lstrip(".")
        )

    @staticmethod
    def _extract_docx(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract text from DOCX files."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx required for DOCX support: pip install python-docx")
        
        text_parts = []
        try:
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    if row_text.strip():
                        text_parts.append(row_text)
        except Exception as e:
            raise ValueError(f"Failed to extract DOCX: {e}")
        
        return ExtractedContent(
            text="\n\n".join(text_parts),
            metadata={
                "file_name": file_path.name,
                "file_type": "docx",
            },
            file_type="docx"
        )

    @staticmethod
    def _extract_doc(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract text from legacy DOC files."""
        try:
            from docx2docx import convert
        except ImportError:
            raise ImportError("python-docx required for DOC support: pip install python-docx")
        
        # Convert DOC to DOCX first
        try:
            docx_content = convert(io.BytesIO(content))
            return FileExtractor._extract_docx(file_path, docx_content)
        except Exception as e:
            raise ValueError(f"Failed to extract DOC: {e}")

    @staticmethod
    def _extract_csv(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract and format CSV data."""
        try:
            import csv
        except ImportError:
            raise ImportError("csv module required")
        
        text_parts = []
        try:
            csv_content = content.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(csv_content))
            
            for idx, row in enumerate(reader):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                text_parts.append(row_text)
        except Exception as e:
            raise ValueError(f"Failed to extract CSV: {e}")
        
        return ExtractedContent(
            text="\n".join(text_parts),
            metadata={
                "file_name": file_path.name,
                "file_type": "csv",
            },
            file_type="csv"
        )

    @staticmethod
    def _extract_excel(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract data from Excel files."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl required for Excel support: pip install openpyxl")
        
        text_parts = []
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(content))
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"--- Sheet: {sheet_name} ---")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                    if row_text.strip():
                        text_parts.append(row_text)
                
                text_parts.append("")
        except Exception as e:
            raise ValueError(f"Failed to extract Excel: {e}")
        
        return ExtractedContent(
            text="\n".join(text_parts),
            metadata={
                "file_name": file_path.name,
                "file_type": "excel",
            },
            file_type="excel"
        )

    @staticmethod
    def _extract_image(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract text from images using OCR."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pytesseract and pillow required for image support: pip install pytesseract pillow")
        
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                text = "[Image processed but no text detected via OCR]"
        except Exception as e:
            raise ValueError(f"Failed to extract from image: {e}")
        
        return ExtractedContent(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_type": "image",
                "format": file_path.suffix.lower().lstrip("."),
            },
            file_type="image"
        )

    @staticmethod
    def _extract_video(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract metadata and transcript from video files."""
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
        except ImportError:
            raise ImportError("moviepy required for video support: pip install moviepy")
        
        text_parts = []
        metadata_dict = {}
        
        try:
            with open("_temp_video.tmp", "wb") as f:
                f.write(content)
            
            clip = VideoFileClip("_temp_video.tmp")
            metadata_dict = {
                "duration": str(clip.duration),
                "fps": str(clip.fps),
                "size": f"{clip.w}x{clip.h}",
            }
            
            if clip.audio is not None:
                # Placeholder: In production, use speech-to-text service
                text_parts.append("[Audio track present - requires speech-to-text transcription]")
            
            clip.close()
            Path("_temp_video.tmp").unlink(missing_ok=True)
        except Exception as e:
            Path("_temp_video.tmp").unlink(missing_ok=True)
            raise ValueError(f"Failed to extract from video: {e}")
        
        return ExtractedContent(
            text="\n".join(text_parts) or "[Video file processed]",
            metadata={
                "file_name": file_path.name,
                "file_type": "video",
                **metadata_dict,
            },
            file_type="video"
        )

    @staticmethod
    def _extract_audio(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract transcript from audio files."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub required for audio support: pip install pydub")
        
        # Placeholder: In production, use speech-to-text service (Deepgram, Whisper, etc.)
        return ExtractedContent(
            text="[Audio file processed - requires speech-to-text transcription]",
            metadata={
                "file_name": file_path.name,
                "file_type": "audio",
                "format": file_path.suffix.lower().lstrip("."),
            },
            file_type="audio"
        )
