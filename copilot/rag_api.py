"""RAG management API handlers for the copilot dashboard."""

from __future__ import annotations

import logging
from typing import Optional

from aiohttp import web

from .rag_integration import CopilotRAGRetriever

LOGGER = logging.getLogger("copilot.rag_api")


class RAGAPIHandler:
    """REST API endpoints for RAG knowledge management."""
    
    def __init__(self, rag_retriever: CopilotRAGRetriever):
        self.rag = rag_retriever
    
    async def upload_knowledge_file(self, request: web.Request) -> web.Response:
        """Upload knowledge file to tenant's knowledge base.
        
        POST /api/rag/upload
        multipart/form-data:
          - file: binary file content
          - tenant_id: string (required)
        
        Returns:
            JSON response with upload status
        """
        try:
            reader = await request.multipart()
            file_data = None
            file_name = None
            tenant_id = None
            
            async for field in reader:
                if field.name == "file":
                    file_name = field.filename
                    file_data = await field.read()
                elif field.name == "tenant_id":
                    tenant_id = (await field.read()).decode().strip()
            
            if not file_data or not file_name or not tenant_id:
                return web.json_response(
                    {"error": "Missing file or tenant_id"},
                    status=400
                )
            
            result = self.rag.upload_knowledge(
                file_path=file_name,
                tenant_id=tenant_id,
                file_content=file_data,
            )
            
            return web.json_response(result, status=200 if result["status"] == "success" else 400)
        
        except Exception as e:
            LOGGER.error(f"Upload failed: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def delete_knowledge_file(self, request: web.Request) -> web.Response:
        """Delete a knowledge file from tenant's knowledge base.
        
        DELETE /api/rag/file/{file_name}?tenant_id={tenant_id}
        
        Returns:
            JSON response with deletion status
        """
        try:
            file_name = request.match_info.get("file_name")
            tenant_id = request.query.get("tenant_id")
            
            if not file_name or not tenant_id:
                return web.json_response(
                    {"error": "Missing file_name or tenant_id"},
                    status=400
                )
            
            result = self.rag.delete_knowledge_file(
                file_name=file_name,
                tenant_id=tenant_id,
            )
            
            return web.json_response(result, status=200 if result["status"] == "success" else 400)
        
        except Exception as e:
            LOGGER.error(f"Deletion failed: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def list_knowledge_files(self, request: web.Request) -> web.Response:
        """List all knowledge files for a tenant.
        
        GET /api/rag/files?tenant_id={tenant_id}
        
        Returns:
            JSON response with file list
        """
        try:
            tenant_id = request.query.get("tenant_id")
            if not tenant_id:
                return web.json_response(
                    {"error": "Missing tenant_id"},
                    status=400
                )
            
            result = self.rag.list_knowledge_files(tenant_id)
            
            return web.json_response(result, status=200 if result["status"] == "success" else 400)
        
        except Exception as e:
            LOGGER.error(f"List failed: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def delete_tenant(self, request: web.Request) -> web.Response:
        """Delete all knowledge for a tenant (admin only).
        
        DELETE /api/rag/tenant/{tenant_id}
        
        Returns:
            JSON response with deletion status
        """
        try:
            tenant_id = request.match_info.get("tenant_id")
            
            if not tenant_id:
                return web.json_response(
                    {"error": "Missing tenant_id"},
                    status=400
                )
            
            result = self.rag.delete_tenant(tenant_id)
            
            return web.json_response(result, status=200 if result["status"] == "success" else 400)
        
        except Exception as e:
            LOGGER.error(f"Tenant deletion failed: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def get_rag_status(self, request: web.Request) -> web.Response:
        """Get RAG system status.
        
        GET /api/rag/status
        
        Returns:
            JSON response with system status
        """
        return web.json_response({
            "status": "ok",
            "rag_enabled": True,
            "admin_mode": self.rag.admin_mode,
            "embedding_model": "text-embedding-3-small",
            "pinecone_indexes": ["admin-kb", "subscriber-kb"],
        })
