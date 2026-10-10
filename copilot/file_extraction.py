"""Multi-format file extraction utilities for RAG system."""

from __future__ import annotations

import io
import mimetypes
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF for PDFs
except Exception:
    fitz = None


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
        global fitz
        if fitz is None:
            try:
                import fitz
            except Exception:
                fitz = None
        if fitz is None:
            raise ValueError("PyMuPDF (fitz) is not available on this environment.")
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
        """Extract text from images using OCR, with Groq Vision fallback."""
        text = ""
        # Try local Tesseract OCR first
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
        except Exception:
            pass
            
        if not text.strip():
            # Fallback to Groq Vision
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    import base64
                    from groq import Groq
                    base64_image = base64.b64encode(content).decode('utf-8')
                    suffix = file_path.suffix.lower()
                    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
                    client = Groq(api_key=groq_key)
                    response = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract all text, numbers, lists, and visible information from this image. Output only the extracted text exactly as it appears. Do not summarize, format as markdown, or add conversational text."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.1,
                        max_completion_tokens=2000
                    )
                    text = response.choices[0].message.content or ""
                except Exception as vision_exc:
                    text = f"[Image processing failed: {vision_exc}]"
            else:
                text = "[Image processed but no text detected via OCR. Set GROQ_API_KEY to enable cloud vision OCR.]"
                
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
    def _transcribe_audio_via_deepgram(content: bytes, suffix: str) -> str:
        """Transcribe audio/video content using Deepgram pre-recorded API."""
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_key:
            return "[Media file processed - Deepgram key not found in environment]"
            
        # Map mime type
        if suffix == ".mp3":
            mime = "audio/mpeg"
        elif suffix == ".wav":
            mime = "audio/wav"
        elif suffix == ".mp4":
            mime = "video/mp4"
        elif suffix == ".m4a":
            mime = "audio/mp4"
        elif suffix in (".mov",):
            mime = "video/quicktime"
        elif suffix in (".webm",):
            mime = "video/webm"
        elif suffix == ".ogg":
            mime = "audio/ogg"
        else:
            mime = "application/octet-stream"
            
        url = "https://api.deepgram.com/v1/listen?smart_format=true&model=nova-3"
        headers = {
            "Authorization": f"Token {deepgram_key}",
            "Content-Type": mime
        }
        
        try:
            resp = requests.post(url, headers=headers, data=content, timeout=90.0)
            if resp.status_code != 200:
                return f"[Media transcription failed: HTTP {resp.status_code} - {resp.text}]"
            result = resp.json()
            return result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        except Exception as e:
            return f"[Media transcription failed: {e}]"

    @staticmethod
    def _extract_video(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract transcript from video files using Deepgram API."""
        text = FileExtractor._transcribe_audio_via_deepgram(content, file_path.suffix.lower())
        return ExtractedContent(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_type": "video",
                "format": file_path.suffix.lower().lstrip("."),
            },
            file_type="video"
        )

    @staticmethod
    def _extract_audio(file_path: Path, content: bytes) -> ExtractedContent:
        """Extract transcript from audio files using Deepgram API."""
        text = FileExtractor._transcribe_audio_via_deepgram(content, file_path.suffix.lower())
        return ExtractedContent(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_type": "audio",
                "format": file_path.suffix.lower().lstrip("."),
            },
            file_type="audio"
        )
