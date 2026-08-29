"""Tests for Pinecone RAG system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from copilot.rag_pinecone import PineconeRAG, RAGChunk


class TestPineconeRAG:
    """Test Pinecone RAG system."""
    
    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client."""
        with patch("copilot.rag_pinecone.Pinecone") as mock:
            yield mock
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("copilot.rag_pinecone.openai") as mock:
            yield mock
    
    def test_initialization(self, mock_pinecone, mock_openai):
        """Test RAG initialization."""
        rag = PineconeRAG(
            pinecone_api_key="test_key",
            openai_api_key="test_openai",
            index_name="subscriber-kb",
        )
        
        assert rag.index_name == "subscriber-kb"
    
    def test_admin_mode_initialization(self, mock_pinecone, mock_openai):
        """Test RAG initialization in admin mode."""
        rag = PineconeRAG(
            pinecone_api_key="test_key",
            openai_api_key="test_openai",
            index_name="admin-kb",
        )
        
        assert rag.index_name == "admin-kb"
    
    def test_generate_chunk_id(self, mock_pinecone, mock_openai):
        """Test deterministic chunk ID generation."""
        rag = PineconeRAG(
            pinecone_api_key="test_key",
            openai_api_key="test_openai",
        )
        
        id1 = rag._generate_chunk_id("tenant1", "file.pdf", 0)
        id2 = rag._generate_chunk_id("tenant1", "file.pdf", 0)
        id3 = rag._generate_chunk_id("tenant1", "file.pdf", 1)
        
        assert id1 == id2  # Same inputs
        assert id1 != id3  # Different chunk index
    
    def test_split_into_chunks(self, mock_pinecone, mock_openai):
        """Test text chunking."""
        rag = PineconeRAG(
            pinecone_api_key="test_key",
            openai_api_key="test_openai",
        )
        
        text = "a" * 5000  # 5000 character text
        chunks = rag._split_into_chunks(text, chunk_size=1500, overlap=150)
        
        assert len(chunks) > 1
        assert all(len(c) <= 1500 for c in chunks)


class TestRAGChunk:
    """Test RAG chunk data structure."""
    
    def test_rag_chunk_creation(self):
        """Test creating RAG chunk."""
        chunk = RAGChunk(
            id="chunk_1",
            text="Sample text",
            source="file.pdf",
            tenant_id="tenant_1",
            file_type="pdf",
            metadata={"page": "1"},
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.source == "file.pdf"
        assert chunk.tenant_id == "tenant_1"
        assert len(chunk.embedding) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
