"""
Snowflake utilities for RAG (Retrieval-Augmented Generation) knowledge base
"""
import json
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from snowflake.connector import connect
from snowflake.connector.errors import ProgrammingError
import logging

logger = logging.getLogger(__name__)


class SnowflakeManager:
    """Manages Snowflake connections and operations for RAG knowledge base"""
    
    def __init__(self, connection_params: Dict[str, str], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Snowflake manager
        
        Args:
            connection_params: Dictionary with connection parameters (account, user, password, warehouse, database, schema)
            embedding_model: Model to use for embeddings (Snowflake Cortex AI model)
        """
        self.connection_params = connection_params
        self.embedding_model = embedding_model
        self.conn = None
        self.cursor = None
    
    def connect(self) -> bool:
        """
        Connect to Snowflake
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.conn = connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            logger.info("Successfully connected to Snowflake")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False
    
    def setup_database(self) -> bool:
        """
        Setup database tables for RAG knowledge base
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Create tables for storing documents and chunks - use quoted identifiers to preserve case
            query1 = 'CREATE TABLE IF NOT EXISTS "documents" ("document_id" VARCHAR(36) PRIMARY KEY, "team_id" VARCHAR(255) NOT NULL, "filename" VARCHAR(1024) NOT NULL, "file_size" INT NOT NULL, "num_chunks" INT NOT NULL, "uploaded_at" TIMESTAMP_NTZ NOT NULL, "created_at" TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())'
            query2 = 'CREATE TABLE IF NOT EXISTS "document_chunks" ("chunk_id" VARCHAR(36) PRIMARY KEY, "document_id" VARCHAR(36) NOT NULL, "team_id" VARCHAR(255) NOT NULL, "chunk_text" VARCHAR(4000) NOT NULL, "chunk_index" NUMBER(38,0) NOT NULL, "embedding" VECTOR(FLOAT, 768), "created_at" TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(), FOREIGN KEY ("document_id") REFERENCES "documents"("document_id"))'
            
            self.cursor.execute(query1)
            self.cursor.execute(query2)
            
            self.conn.commit()
            logger.info("Database tables setup successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return False
    
    def insert_document_metadata(
        self,
        document_id: str,
        team_id: str,
        filename: str,
        file_size: int,
        num_chunks: int
    ) -> bool:
        """
        Insert document metadata into database
        
        Args:
            document_id: Unique document ID
            team_id: Team ID for access control
            filename: Original filename
            file_size: File size in bytes/characters
            num_chunks: Number of chunks created from document
            
        Returns:
            True if insert successful, False otherwise
        """
        try:
            query = 'INSERT INTO "documents" ("document_id", "team_id", "filename", "file_size", "num_chunks", "uploaded_at") VALUES (%s, %s, %s, %s, %s, %s)'
            
            self.cursor.execute(query, (
                document_id,
                team_id,
                filename,
                file_size,
                num_chunks,
                datetime.utcnow().isoformat()
            ))
            
            self.conn.commit()
            logger.info(f"Inserted document metadata for {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert document metadata: {e}")
            return False
    
    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        team_id: str,
        document_id: str
    ) -> bool:
        """
        Insert document chunks with embeddings into database
        
        Args:
            chunks: List of chunk dictionaries with 'text' and optionally 'embedding'
            team_id: Team ID for access control
            document_id: Document ID that chunks belong to
            
        Returns:
            True if insert successful, False otherwise
        """
        try:
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_text = chunk.get("text", "")
                # Validate and truncate chunk_text to 4000 characters to match table definition
                if len(chunk_text) > 4000:
                    logger.warning(f"Chunk {chunk_id} text exceeds 4000 chars ({len(chunk_text)}), truncating")
                    chunk_text = chunk_text[:4000]
                
                try:
                    # Insert with embedding using INSERT...SELECT format (required for function calls with VECTOR type)
                    query = 'INSERT INTO "document_chunks" ("chunk_id", "document_id", "team_id", "chunk_text", "chunk_index", "embedding") SELECT %s, %s, %s, %s, %s, SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s)'
                    
                    self.cursor.execute(query, (
                        chunk_id,
                        document_id,
                        team_id,
                        chunk_text,
                        idx,
                        'snowflake-arctic-embed-m-v1.5',
                        chunk_text
                    ))
                except Exception as embed_error:
                    logger.warning(f"Failed to insert with embedding, inserting without: {embed_error}")
                    # Insert without embedding if embedding generation failed
                    query_no_embed = 'INSERT INTO "document_chunks" ("chunk_id", "document_id", "team_id", "chunk_text", "chunk_index") SELECT %s, %s, %s, %s, %s'
                    self.cursor.execute(query_no_embed, (
                        chunk_id,
                        document_id,
                        team_id,
                        chunk_text,
                        idx
                    ))
            
            self.conn.commit()
            logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            return False
    
    def _get_embedding(self, text: str) -> Optional[Any]:
        """
        Get embedding for text using Snowflake Cortex AI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector object or None
        """
        try:
            # Use Snowflake Cortex AI to generate embeddings
            # Using snowflake-arctic-embed-m-v1.5 model which is optimized for semantic search
            query = "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', %s)"
            
            self.cursor.execute(query, (text,))
            result = self.cursor.fetchone()
            
            if result and result[0]:
                # Return the embedding vector object directly (not as string)
                return result[0]
            else:
                # Return None if failed
                logger.warning(f"Failed to get embedding for text: {text[:50]}")
                return None
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return None on error
            return None
    
    def get_team_documents(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a team
        
        Args:
            team_id: Team ID to retrieve documents for
            
        Returns:
            List of document metadata dictionaries
        """
        try:
            query = 'SELECT "document_id", "filename", "file_size", "num_chunks", "uploaded_at" FROM "documents" WHERE "team_id" = %s ORDER BY "uploaded_at" DESC'
            
            self.cursor.execute(query, (team_id,))
            results = self.cursor.fetchall()
            
            documents = []
            for row in results:
                documents.append({
                    "document_id": row[0],
                    "filename": row[1],
                    "file_size": row[2],
                    "num_chunks": row[3],
                    "uploaded_at": row[4].isoformat() if hasattr(row[4], 'isoformat') else str(row[4])
                })
            
            return documents
        except Exception as e:
            logger.error(f"Failed to get team documents: {e}")
            return []
    
    def query_with_cortex_ai(
        self,
        query: str,
        team_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query knowledge base using semantic search and generate answer with Cortex AI
        
        Args:
            query: User's question
            team_id: Team ID for access control
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        try:
            # Search for relevant chunks using vector similarity
            # Generate embedding inline in the SQL statement
            search_query = 'SELECT c."chunk_id", c."chunk_text", d."filename", VECTOR_COSINE_SIMILARITY(c."embedding", SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s)) as similarity FROM "document_chunks" c JOIN "documents" d ON c."document_id" = d."document_id" WHERE c."team_id" = %s AND c."embedding" IS NOT NULL ORDER BY similarity DESC LIMIT %s'
            
            self.cursor.execute(search_query, ('snowflake-arctic-embed-m-v1.5', query, team_id, top_k))
            chunks = self.cursor.fetchall()
            
            if not chunks:
                return {
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": []
                }
            
            # Prepare context from retrieved chunks
            context = "\n\n".join([f"From {chunk[2]}:\n{chunk[1]}" for chunk in chunks])
            
            # Generate answer using Snowflake Cortex AI
            prompt = f"""Based on the following context from our knowledge base, answer the user's question.
            
Context:
{context}

Question: {query}

Answer:"""
            
            # Use Cortex AI for generating answer
            ai_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', %s)"
            
            self.cursor.execute(ai_query, (prompt,))
            result = self.cursor.fetchone()
            
            answer = result[0] if result else "Unable to generate answer"
            
            sources = [
                {
                    "chunk_id": chunk[0],
                    "text": chunk[1],
                    "filename": chunk[2],
                    "similarity": float(chunk[3])
                }
                for chunk in chunks
            ]
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Failed to query with Cortex AI: {e}")
            return {
                "answer": f"Error querying knowledge base: {str(e)}",
                "sources": []
            }
    
    def close(self):
        """Close Snowflake connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Closed Snowflake connection")
