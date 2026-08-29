"""Pinecone-based RAG system for multi-tenant SaaS platform."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import openai
from pinecone import Pinecone

from .file_extraction import ExtractedContent, FileExtractor

LOGGER = logging.getLogger("copilot.rag")

# Pinecone configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100


@dataclass(frozen=True, slots=True)
class RAGChunk:
    """Vector store chunk with metadata."""
    id: str
    text: str
    source: str
    tenant_id: str
    file_type: str
    metadata: dict[str, str]
    embedding: list[float]


class PineconeRAG:
    """Multi-tenant RAG system using Pinecone and OpenAI embeddings."""

    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str = "subscriber-kb",
    ):
        """Initialize Pinecone RAG.
        
        Args:
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: The single Pinecone index name to use
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.index_name = index_name
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Get or create Pinecone index."""
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            LOGGER.info(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        
        return self.pc.Index(self.index_name)

    def _get_embedding(self, text: str) -> list[float]:
        """Get OpenAI embedding for text."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def _generate_chunk_id(self, tenant_id: str, file_name: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        hash_input = f"{tenant_id}#{file_name}#{chunk_index}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _split_into_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 150) -> list[str]:
        """Split text into overlapping chunks for better retrieval."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks if chunks else [text]

    def upload_file(
        self,
        file_path: str,
        tenant_id: str,
        file_content: Optional[bytes] = None,
        chunk_size: int = 1500,
        scope: str = "sales",
        owner_id: Optional[str] = None,
    ) -> dict[str, int]:
        """Upload and index file content.
        
        Args:
            file_path: Path to file
            tenant_id: Tenant identifier (namespace)
            file_content: Binary file content (optional, reads from disk if not provided)
            chunk_size: Size of text chunks
            scope: The scope of the document ("sales" or "admin")
            owner_id: Optional owner identifier (individual user)
            
        Returns:
            Dict with upload stats (chunks_uploaded, total_size)
        """
        try:
            extracted = FileExtractor.extract(file_path, file_content)
        except Exception as e:
            LOGGER.error(f"Failed to extract {file_path}: {e}")
            raise

        chunks = self._split_into_chunks(extracted.text, chunk_size=chunk_size)
        vectors_to_upsert = []
        
        file_name = extracted.metadata.get("file_name", "unknown")
        
        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            chunk_id = self._generate_chunk_id(tenant_id, file_name, chunk_idx)
            embedding = self._get_embedding(chunk_text)
            
            vectors_to_upsert.append((
                chunk_id,
                embedding,
                {
                    "text": chunk_text,
                    "source": file_name,
                    "file_type": extracted.file_type,
                    "tenant_id": tenant_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "scope": scope,
                    "owner_id": owner_id or "",
                    **extracted.metadata,
                }
            ))
        
        # Batch upsert to Pinecone
        batch_count = 0
        for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
            batch = vectors_to_upsert[i:i + BATCH_SIZE]
            self.index.upsert(
                vectors=batch,
                namespace=tenant_id
            )
            batch_count += len(batch)
            LOGGER.info(f"Uploaded {batch_count}/{len(vectors_to_upsert)} chunks")
        
        return {
            "chunks_uploaded": len(vectors_to_upsert),
            "total_size": len(extracted.text),
            "file_type": extracted.file_type,
        }

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        min_score: float = 0.5,
        scope: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> list[RAGChunk]:
        """Search for relevant chunks using vector similarity.
        
        Args:
            query: Search query
            tenant_id: Tenant identifier (namespace)
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            scope: Optional metadata scope filter
            owner_id: Optional metadata owner filter
            
        Returns:
            List of relevant RAG chunks
        """
        try:
            query_embedding = self._get_embedding(query)
        except Exception as e:
            LOGGER.error(f"Failed to embed query: {e}")
            return []
        
        filter_dict = {}
        if scope:
            filter_dict["scope"] = {"$eq": scope}
        if owner_id:
            filter_dict["owner_id"] = {"$eq": owner_id}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=tenant_id,
            include_metadata=True,
            include_values=False,
            filter=filter_dict if filter_dict else None,
        )
        
        chunks = []
        for match in results.get("matches", []):
            if match["score"] < min_score:
                continue
            
            metadata = match.get("metadata", {})
            chunks.append(RAGChunk(
                id=match["id"],
                text=metadata.get("text", ""),
                source=metadata.get("source", "unknown"),
                tenant_id=metadata.get("tenant_id", tenant_id),
                file_type=metadata.get("file_type", "unknown"),
                metadata={k: v for k, v in metadata.items() 
                         if k not in {"text", "source", "file_type", "tenant_id"}},
                embedding=[],  # Not included in query response
            ))
        
        return chunks

    def delete_tenant_data(self, tenant_id: str) -> dict[str, int]:
        """Delete all data for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dict with deletion stats
        """
        # Pinecone doesn't have native namespace deletion,
        # so we query all vectors in namespace and delete them
        try:
            results = self.index.query(
                vector=[0.0] * EMBEDDING_DIMENSION,
                top_k=10000,
                namespace=tenant_id,
                include_metadata=False,
            )
            
            ids_to_delete = [match["id"] for match in results.get("matches", [])]
            
            if ids_to_delete:
                self.index.delete(
                    ids=ids_to_delete,
                    namespace=tenant_id
                )
            
            LOGGER.info(f"Deleted {len(ids_to_delete)} chunks for tenant {tenant_id}")
            return {"deleted_count": len(ids_to_delete)}
        except Exception as e:
            LOGGER.error(f"Failed to delete tenant data: {e}")
            raise

    def delete_file(self, file_name: str, tenant_id: str) -> dict[str, int]:
        """Delete specific file from tenant's knowledge base.
        
        Args:
            file_name: Name of file to delete
            tenant_id: Tenant identifier
            
        Returns:
            Dict with deletion stats
        """
        # Query all chunks from this file
        try:
            results = self.index.query(
                vector=[0.0] * EMBEDDING_DIMENSION,
                top_k=10000,
                namespace=tenant_id,
                include_metadata=True,
                filter={"source": {"$eq": file_name}}
            )
            
            ids_to_delete = [match["id"] for match in results.get("matches", [])]
            
            if ids_to_delete:
                self.index.delete(
                    ids=ids_to_delete,
                    namespace=tenant_id
                )
            
            LOGGER.info(f"Deleted {len(ids_to_delete)} chunks for file {file_name}")
            return {"deleted_count": len(ids_to_delete)}
        except Exception as e:
            LOGGER.error(f"Failed to delete file: {e}")
            raise

    def list_tenant_files(self, tenant_id: str) -> list[dict[str, str]]:
        """List all uploaded files for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of file metadata dicts
        """
        try:
            # Query all chunks in namespace
            results = self.index.query(
                vector=[0.0] * EMBEDDING_DIMENSION,
                top_k=10000,
                namespace=tenant_id,
                include_metadata=True,
            )
            
            files_dict = {}
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                source = metadata.get("source", "unknown")
                
                if source not in files_dict:
                    files_dict[source] = {
                        "file_name": source,
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunk_count": 0,
                    }
                files_dict[source]["chunk_count"] += 1
            
            return list(files_dict.values())
        except Exception as e:
            LOGGER.error(f"Failed to list tenant files: {e}")
            return []
