"""Integration layer between Twilio copilot and Pinecone RAG."""

from __future__ import annotations

import logging
from typing import Optional, Callable

from .rag_pinecone import PineconeRAG, RAGChunk

LOGGER = logging.getLogger("copilot.rag_integration")


class TenantContext:
    """Context information for a tenant."""
    
    def __init__(self, tenant_id: str, is_admin: bool = False):
        self.tenant_id = tenant_id
        self.is_admin = is_admin


class CopilotRAGRetriever:
    """Retrieves knowledge from Pinecone for copilot suggestions.
    
    This replaces the LocalKnowledgeBase for real-time call context.
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        pinecone_index_name: str = "subscriber-kb",
    ):
        self.rag = PineconeRAG(
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key,
            index_name=pinecone_index_name,
        )
    
    def get_context(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3,
        min_score: float = 0.5,
        scope: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> list[str]:
        """Get knowledge context for a query.
        
        Args:
            query: Client's question or statement
            tenant_id: Tenant namespace
            top_k: Number of chunks to retrieve
            min_score: Minimum vector similarity score
            scope: Optional metadata scope filter
            owner_id: Optional metadata owner filter
            
        Returns:
            List of relevant knowledge chunks as strings
        """
        try:
            chunks = self.rag.search(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k,
                min_score=min_score,
                scope=scope,
                owner_id=owner_id,
            )
            
            formatted_chunks = []
            for chunk in chunks:
                formatted = f"From {chunk.source}:\n{chunk.text}"
                formatted_chunks.append(formatted)
                LOGGER.debug(f"Retrieved chunk: {chunk.id} (score available in vector DB)")
            
            return formatted_chunks
        except Exception as e:
            LOGGER.error(f"RAG retrieval failed for tenant {tenant_id}: {e}")
            return []
    
    def upload_knowledge(
        self,
        file_path: str,
        tenant_id: str,
        file_content: Optional[bytes] = None,
        scope: str = "sales",
        owner_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Upload knowledge file for a tenant.
        
        Args:
            file_path: Path to file (or just filename if using file_content)
            tenant_id: Tenant namespace
            file_content: Binary file content
            scope: The scope of the document ("sales" or "admin")
            owner_id: Optional owner identifier
            progress_callback: Optional progress callback
            
        Returns:
            Upload status dict
        """
        try:
            result = self.rag.upload_file(
                file_path=file_path,
                tenant_id=tenant_id,
                file_content=file_content,
                scope=scope,
                owner_id=owner_id,
                progress_callback=progress_callback,
            )
            
            LOGGER.info(f"Uploaded {file_path} for tenant {tenant_id}: {result}")
            return {
                "status": "success",
                **result
            }
        except Exception as e:
            LOGGER.error(f"Upload failed for {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def delete_knowledge_file(
        self,
        file_name: str,
        tenant_id: str,
    ) -> dict:
        """Delete a knowledge file.
        
        Args:
            file_name: Name of file to delete
            tenant_id: Tenant namespace
            
        Returns:
            Deletion status dict
        """
        try:
            result = self.rag.delete_file(
                file_name=file_name,
                tenant_id=tenant_id,
            )
            
            LOGGER.info(f"Deleted {file_name} for tenant {tenant_id}")
            return {
                "status": "success",
                **result
            }
        except Exception as e:
            LOGGER.error(f"Deletion failed for {file_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def list_knowledge_files(self, tenant_id: str) -> dict:
        """List uploaded knowledge files for a tenant.
        
        Args:
            tenant_id: Tenant namespace
            
        Returns:
            Dict with file list and stats
        """
        try:
            files = self.rag.list_tenant_files(tenant_id)
            return {
                "status": "success",
                "tenant_id": tenant_id,
                "files": files,
                "total_files": len(files),
            }
        except Exception as e:
            LOGGER.error(f"Failed to list files for tenant {tenant_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def delete_tenant(self, tenant_id: str) -> dict:
        """Delete all knowledge for a tenant (admin only).
        
        Args:
            tenant_id: Tenant namespace
            
        Returns:
            Deletion status dict
        """
        if not self.admin_mode:
            return {
                "status": "error",
                "error": "Only admin can delete tenant data"
            }
        
        try:
            result = self.rag.delete_tenant_data(tenant_id)
            LOGGER.info(f"Deleted all data for tenant {tenant_id}")
            return {
                "status": "success",
                **result
            }
        except Exception as e:
            LOGGER.error(f"Tenant deletion failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
