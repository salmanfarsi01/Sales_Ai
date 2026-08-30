"""Tests for image OCR and audio/video transcription fallbacks in FileExtractor."""

import os
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path
from copilot.file_extraction import FileExtractor, ExtractedContent


@patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
@patch("groq.Groq")
def test_extract_image_groq_vision_fallback(mock_groq):
    """Test that image extraction falls back to Groq Vision when Tesseract fails."""
    # Mock Tesseract to raise exception
    with patch("pytesseract.image_to_string", side_effect=Exception("Tesseract not found")):
        # Mock PIL image loading
        with patch("PIL.Image.open") as mock_image_open:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client
            
            # Mock Groq API response
            mock_message = MagicMock()
            mock_message.content = "Extracted text from image"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_client.chat.completions.create.return_value.choices = [mock_choice]
            
            result = FileExtractor.extract("test.png", file_content=b"fake_image_bytes")
            
            assert isinstance(result, ExtractedContent)
            assert result.text == "Extracted text from image"
            assert result.file_type == "image"
            assert result.metadata["format"] == "png"
            
            # Verify Groq vision was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            called_kwargs = mock_client.chat.completions.create.call_args[1]
            assert called_kwargs["model"] == "llama-3.2-11b-vision-preview"


@patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake_deepgram_key"})
@patch("requests.post")
def test_extract_audio_deepgram(mock_post):
    """Test that audio extraction queries Deepgram pre-recorded API successfully."""
    # Mock Deepgram JSON response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello, this is a transcribed audio file."
                        }
                    ]
                }
            ]
        }
    }
    mock_post.return_value = mock_response
    
    result = FileExtractor.extract("test.mp3", file_content=b"fake_audio_bytes")
    
    assert isinstance(result, ExtractedContent)
    assert result.text == "Hello, this is a transcribed audio file."
    assert result.file_type == "audio"
    assert result.metadata["format"] == "mp3"
    
    # Verify requests.post was called with correct headers
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.deepgram.com/v1/listen?smart_format=true&model=nova-3"
    assert kwargs["headers"]["Authorization"] == "Token fake_deepgram_key"
    assert kwargs["headers"]["Content-Type"] == "audio/mpeg"


@patch.dict(os.environ, {"DEEPGRAM_API_KEY": "fake_deepgram_key"})
@patch("requests.post")
def test_extract_video_deepgram(mock_post):
    """Test that video extraction queries Deepgram pre-recorded API successfully."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello, this is a transcribed video file."
                        }
                    ]
                }
            ]
        }
    }
    mock_post.return_value = mock_response
    
    result = FileExtractor.extract("test.mp4", file_content=b"fake_video_bytes")
    
    assert isinstance(result, ExtractedContent)
    assert result.text == "Hello, this is a transcribed video file."
    assert result.file_type == "video"
    assert result.metadata["format"] == "mp4"
    
    # Verify request headers
    mock_post.assert_called_once()
    kwargs = mock_post.call_args[1]
    assert kwargs["headers"]["Content-Type"] == "video/mp4"
