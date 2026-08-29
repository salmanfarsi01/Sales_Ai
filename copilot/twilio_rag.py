"""
Example: RAG-Integrated Twilio Copilot

This shows how to modify twilio_fast.py to use RAG instead of LocalKnowledgeBase.
Apply these changes to your actual twilio_fast.py or create RAGTwilioCopilot class.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiohttp import web
from groq import Groq

from .config_rag import Settings
from .rag_integration import CopilotRAGRetriever
from .rag_api import RAGAPIHandler
from .twilio_fast import FastTwilioCopilot

LOGGER = logging.getLogger("copilot.twilio_rag")
STATIC = Path(__file__).resolve().parent.parent / "web"


class RAGTwilioCopilot(FastTwilioCopilot):
    """FastTwilioCopilot with RAG-powered knowledge retrieval."""
    
    def __init__(self, settings: Settings):
        # Initialize parent (handles Twilio, Deepgram, Groq)
        super().__init__(settings)
        
        # Replace local knowledge base with RAG
        if settings.rag and settings.rag.rag_enabled:
            LOGGER.info("Initializing Pinecone RAG system")
            self.rag_retriever = CopilotRAGRetriever(
                pinecone_api_key=settings.rag.pinecone_api_key,
                openai_api_key=settings.rag.openai_api_key,
                admin_mode=settings.rag.admin_mode,
            )
            # Remove local knowledge base
            self.knowledge = None
            self.rag_api_handler = RAGAPIHandler(self.rag_retriever)
        else:
            LOGGER.info("RAG disabled, using local knowledge base")
            self.rag_retriever = None
            # Keep parent's LocalKnowledgeBase
            self.rag_api_handler = None
        
        self.settings = settings
    
    def app(self) -> web.Application:
        """Create app with RAG API routes."""
        # Get parent's app
        app = super().app()
        
        # Add RAG API routes if RAG is enabled
        if self.rag_api_handler:
            app.router.add_post("/api/rag/upload", self.rag_api_handler.upload_knowledge_file)
            app.router.add_delete("/api/rag/file/{file_name}", self.rag_api_handler.delete_knowledge_file)
            app.router.add_get("/api/rag/files", self.rag_api_handler.list_knowledge_files)
            app.router.add_delete("/api/rag/tenant/{tenant_id}", self.rag_api_handler.delete_tenant)
            app.router.add_get("/api/rag/status", self.rag_api_handler.get_rag_status)
        
        return app
    
    async def _get_knowledge_context(
        self,
        query: str,
        tenant_id: str | None = None,
    ) -> str:
        """Get knowledge context for LLM generation.
        
        Uses RAG if enabled, otherwise falls back to local search.
        """
        tenant_id = tenant_id or self.settings.default_tenant_id
        
        if self.rag_retriever:
            # RAG mode: vector search
            context_list = self.rag_retriever.get_context(
                query=query,
                tenant_id=tenant_id,
                top_k=self.settings.rag.search_top_k,
                min_score=self.settings.rag.min_score_threshold,
            )
            return "\n\n".join(context_list)
        else:
            # Local mode: keyword search (backward compatible)
            chunks = self.knowledge.search(query, limit=3)
            return "\n\n".join([f"From {c.source}:\n{c.text}" for c in chunks])
    
    async def _generate_suggestion(
        self,
        question: str,
        transcript: list[dict[str, str]],
        tenant_id: str | None = None,
    ) -> str:
        """Generate suggestion with RAG context.
        
        This overrides the parent method to use RAG for knowledge retrieval.
        """
        # Get knowledge context
        knowledge_context = await self._get_knowledge_context(question, tenant_id)
        
        # Build system prompt
        system_prompt = """You are a helpful sales assistant. Keep responses to 3 concise sentences max.
If relevant knowledge is provided, use it to give specific, actionable advice."""
        
        # Build messages with context
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history
        for turn in transcript:
            role = "user" if turn["speaker"] == "client" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        
        # Add knowledge context and current question
        if knowledge_context:
            context_msg = f"Relevant knowledge:\n{knowledge_context}\n\nCurrent question:\n{question}"
        else:
            context_msg = question
        
        messages.append({"role": "user", "content": context_msg})
        
        # Generate with Groq
        try:
            response = self.groq.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                max_tokens=150,
                stream=True,
                temperature=0.7,
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return full_response
        except Exception as e:
            LOGGER.error(f"Generation failed: {e}")
            return "I encountered an error generating a suggestion."
    
    async def broadcast_rag_status(self) -> None:
        """Broadcast RAG system status to dashboards."""
        if self.rag_retriever:
            status = {
                "type": "rag_status",
                "rag_enabled": True,
                "admin_mode": self.rag_retriever.admin_mode,
            }
        else:
            status = {
                "type": "rag_status",
                "rag_enabled": False,
            }
        
        await self.broadcast(status)


# Backward compatibility
AllQuestionsCopilot = RAGTwilioCopilot
"""
