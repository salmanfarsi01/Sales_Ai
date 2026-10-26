"""Pinecone-based RAG system for multi-tenant SaaS platform."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
        admin_mode: bool = False,
    ):
        """Initialize Pinecone RAG.
        
        Args:
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: The single Pinecone index name to use
            admin_mode: Whether this instance operates in admin mode
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.index_name = index_name
        self.admin_mode = admin_mode
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
            import time
            LOGGER.info("Waiting for Pinecone index to become ready...")
            while True:
                desc = self.pc.describe_index(self.index_name)
                if desc.status.get('ready') or desc.status.get('state') == 'Ready':
                    LOGGER.info(f"Pinecone index {self.index_name} is ready.")
                    break
                time.sleep(2)
        
        return self.pc.Index(self.index_name)

    def _get_embedding(self, text: str, timeout: float = 0.35) -> list[float]:
        """Get OpenAI embedding for text with 350ms timeout."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            timeout=timeout,
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
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, int]:
        """Upload and index file content.
        
        Args:
            file_path: Path to file
            tenant_id: Tenant identifier (namespace)
            file_content: Binary file content (optional, reads from disk if not provided)
            chunk_size: Size of text chunks
            scope: The scope of the document ("sales" or "admin")
            owner_id: Optional owner identifier (individual user)
            progress_callback: Callback triggered with (chunks_uploaded, total_chunks)
            
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
            
            # Align metadata keys: write both (owner_id, owner_user_id) and (scope, owner_scope)
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
                    "owner_scope": scope,
                    "owner_id": owner_id or "",
                    "owner_user_id": owner_id or "",
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
            if progress_callback:
                progress_callback(batch_count, len(vectors_to_upsert))
        
        return {
            "chunks_uploaded": len(vectors_to_upsert),
            "total_size": len(extracted.text),
            "file_type": extracted.file_type,
        }

    def upsert_text_chunk(
        self,
        chunk_id: str,
        text: str,
        tenant_id: str,
        metadata: Optional[dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> bool:
        """Upsert a single text chunk into Pinecone index under tenant namespace."""
        try:
            embedding = self._get_embedding(text, timeout=timeout)
            meta = {
                "text": text,
                "tenant_id": tenant_id,
                "source": metadata.get("source_title", metadata.get("source", "knowledge_record")) if metadata else "knowledge_record",
                "file_type": "text",
                **(metadata or {}),
            }
            self.index.upsert(
                vectors=[(chunk_id, embedding, meta)],
                namespace=tenant_id,
            )
            LOGGER.info("Upserted chunk %s to Pinecone (namespace: %s)", chunk_id, tenant_id)
            return True
        except Exception as e:
            LOGGER.error("Failed to upsert text chunk %s to Pinecone: %s", chunk_id, e)
            return False

    def upsert_record(self, record: Any, tenant_id: Optional[str] = None) -> bool:
        """Upsert a KnowledgeRecord directly into Pinecone index."""
        try:
            ns = tenant_id or getattr(record, "owner_user_id", "default")
            meta = {
                "text": record.text,
                "source": getattr(record, "source_title", "knowledge_record"),
                "source_title": getattr(record, "source_title", "knowledge_record"),
                "source_version": getattr(record, "source_version", "1.0"),
                "knowledge_class": getattr(record, "knowledge_class", "general"),
                "verification_status": getattr(record, "verification_status", "company_approved"),
                "sensitivity": getattr(record, "sensitivity", "normal"),
                "owner_user_id": getattr(record, "owner_user_id", ns),
                "owner_id": getattr(record, "owner_user_id", ns),
                "scope": getattr(record, "owner_scope", "user"),
                "owner_scope": getattr(record, "owner_scope", "user"),
                "topic_tags": getattr(record, "topic_tags", []),
            }
            return self.upsert_text_chunk(
                chunk_id=getattr(record, "knowledge_id", str(uuid.uuid4().hex[:16])),
                text=record.text,
                tenant_id=ns,
                metadata=meta,
            )
        except Exception as e:
            LOGGER.error("Failed to upsert KnowledgeRecord to Pinecone: %s", e)
            return False

    def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        min_score: float = 0.5,
        scope: Optional[str] = None,
        owner_id: Optional[str] = None,
        timeout: float = 5.0,
    ) -> list[RAGChunk]:
        """Search for relevant chunks using vector similarity.
        
        Args:
            query: Search query
            tenant_id: Tenant identifier (namespace)
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            scope: Optional metadata scope filter
            owner_id: Optional metadata owner filter
            timeout: Network call timeout in seconds
            
        Returns:
            List of relevant RAG chunks
        """
        try:
            query_embedding = self._get_embedding(query, timeout=timeout)
        except Exception as e:
            LOGGER.error(f"Failed to embed query: {e}")
            return []
        
        effective_user_id = owner_id
        if not effective_user_id and not self.admin_mode:
            raise ValueError(
                "Mandatory owner filter required: queries within tenant namespace must specify owner_id / owner_user_id (Doc 02 Constraint C6)."
            )

        filter_dict: dict[str, Any] = {}
        if effective_user_id:
            # Enforce exact mandatory isolation filter at Pinecone query level (Doc 02 Constraint C6):
            # owner_user_id == :user AND owner_scope == 'user'
            filter_dict["owner_user_id"] = {"$eq": effective_user_id}
            filter_dict["owner_scope"] = {"$eq": "user"}
        elif scope:
            filter_dict["scope"] = {"$eq": scope}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=tenant_id,
            include_metadata=True,
            include_values=False,
            filter=filter_dict if filter_dict else None,
        )
        
        chunks = []
        LOGGER.info("RAG search query: '%s' namespace: '%s' filter: %s", query, tenant_id, filter_dict)
        LOGGER.info("RAG search results (top_k=%d):", top_k)
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            source = metadata.get("source", "unknown")
            text_snippet = metadata.get("text", "").replace("\n", " ")[:100]
            LOGGER.info("  [%s] Score: %.4f | Source: %s | Snippet: %s...", match["id"], match["score"], source, text_snippet)
            
            # Strict defense-in-depth post-verification: chunk owner must strictly equal effective_user_id
            if effective_user_id:
                chunk_owner = metadata.get("owner_user_id") or metadata.get("owner_id")
                if not chunk_owner or chunk_owner != effective_user_id:
                    LOGGER.warning("Excluding cross-user chunk '%s' lacking matching owner metadata.", match["id"])
                    continue

            if match["score"] < min_score:
                LOGGER.info("     (Skipped: score %.4f below min_score %.4f)", match["score"], min_score)
                continue
            
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

    def migrate_metadata_keys(self, tenant_id: str, batch_size: int = 100) -> dict[str, int]:
        """Migration utility ensuring vectors in a tenant namespace contain both owner_id and owner_user_id, scope and owner_scope."""
        migrated = 0
        total_inspected = 0
        try:
            # Check vectors in namespace
            dummy_vec = [0.0] * EMBEDDING_DIMENSION
            sample = self.index.query(
                vector=dummy_vec,
                top_k=min(batch_size, 100),
                namespace=tenant_id,
                include_metadata=True,
                include_values=False,
            )
            matches = sample.get("matches", [])
            total_inspected = len(matches)
            for m in matches:
                meta = m.get("metadata", {})
                needs_update = False
                updated_meta = dict(meta)
                if "owner_id" in meta and "owner_user_id" not in meta:
                    updated_meta["owner_user_id"] = meta["owner_id"]
                    needs_update = True
                elif "owner_user_id" in meta and "owner_id" not in meta:
                    updated_meta["owner_id"] = meta["owner_user_id"]
                    needs_update = True
                if "scope" in meta and "owner_scope" not in meta:
                    updated_meta["owner_scope"] = meta["scope"]
                    needs_update = True
                elif "owner_scope" in meta and "scope" not in meta:
                    updated_meta["scope"] = meta["owner_scope"]
                    needs_update = True

                if needs_update:
                    self.index.update(
                        id=m["id"],
                        set_metadata=updated_meta,
                        namespace=tenant_id,
                    )
                    migrated += 1
        except Exception as e:
            LOGGER.warning("Metadata migration check encountered: %s", e)

        return {"inspected": total_inspected, "migrated": migrated}

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
