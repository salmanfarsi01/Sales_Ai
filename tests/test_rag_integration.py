"""Tests for RAG integration with copilot."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from copilot.rag_integration import CopilotRAGRetriever, TenantContext


class TestTenantContext:
    """Test tenant context."""
    
    def test_tenant_context_creation(self):
        """Test creating tenant context."""
        ctx = TenantContext("tenant_1", is_admin=False)
        
        assert ctx.tenant_id == "tenant_1"
        assert ctx.is_admin is False
    
    def test_tenant_context_admin(self):
        """Test admin tenant context."""
        ctx = TenantContext("admin", is_admin=True)
        
        assert ctx.tenant_id == "admin"
        assert ctx.is_admin is True


class TestCopilotRAGRetriever:
    """Test copilot RAG retriever."""
    
    @pytest.fixture
    def mock_rag(self):
        """Mock RAG backend."""
        with patch("copilot.rag_integration.PineconeRAG") as mock:
            yield mock
    
    def test_initialization(self, mock_rag):
        """Test retriever initialization."""
        retriever = CopilotRAGRetriever(
            pinecone_api_key="key",
            openai_api_key="key",
            admin_mode=False,
        )
        
        assert retriever.admin_mode is False
        assert retriever.rag is not None
    
    def test_get_context_formatting(self, mock_rag):
        """Test context formatting."""
        # Mock RAG search
        mock_instance = mock_rag.return_value
        mock_instance.search.return_value = [
            Mock(text="Content 1", source="file1.pdf"),
            Mock(text="Content 2", source="file2.md"),
        ]
        
        retriever = CopilotRAGRetriever(
            pinecone_api_key="key",
            openai_api_key="key",
        )
        
        context = retriever.get_context("query", "tenant_1")
        
        assert len(context) == 2
        assert "file1.pdf" in context[0]
        assert "Content 1" in context[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
