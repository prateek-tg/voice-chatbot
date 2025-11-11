"""
Redis Manager for the AI Agentic RAG system.
Handles all Redis operations including session management, caching, and TTL management.
This reduces latency by providing fast in-memory access to conversation history,
cached retrievals, and session state.
"""

import json
import hashlib
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, TimeoutError

from config import REDIS_CONFIG, REDIS_KEYS, REDIS_TTL, ORCHESTRATOR_CONFIG, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Manages all Redis operations for the RAG system.
    
    This class provides fast in-memory storage and retrieval for:
    - Chat sessions and conversation history
    - Intent classifications and session metadata
    - Cached document retrievals and summaries
    - Distributed locks for preventing duplicate operations
    
    The caching strategy significantly reduces latency by:
    1. Avoiding repeated LLM calls for similar queries
    2. Caching expensive ChromaDB retrievals
    3. Maintaining session context without database lookups
    4. Preventing duplicate concurrent operations
    """
    
    def __init__(self, redis_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Redis connection with configuration.
        
        Args:
            redis_config: Optional Redis configuration dictionary
        """
        self.config = redis_config or REDIS_CONFIG
        self.client = None
        self.connect()
    
    def connect(self) -> None:
        """
        Establish connection to Redis server.
        
        Raises:
            ConnectionError: If unable to connect to Redis
        """
        try:
            self.client = redis.Redis(**self.config)
            # Test connection
            self.client.ping()
            logger.info("Successfully connected to Redis")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")
    
    def is_connected(self) -> bool:
        """
        Check if Redis connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            return self.client.ping() if self.client else False
        except:
            return False
    
    # Session Management Methods
    
    def store_chat_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Store a chat message in the session history.
        
        Args:
            session_id: Unique session identifier
            message: Message dictionary with 'role', 'content', 'timestamp'
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            key = REDIS_KEYS["chat_session"].format(session_id=session_id)
            
            # Get existing messages
            existing_messages = self.get_chat_history(session_id) or []
            
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()
            
            # Append new message
            existing_messages.append(message)
            
            # Keep only last N messages (configurable)
            max_messages = 50  # Can be moved to config
            if len(existing_messages) > max_messages:
                existing_messages = existing_messages[-max_messages:]
            
            # Store with TTL
            self.client.setex(
                key,
                REDIS_TTL["chat_session"],
                json.dumps(existing_messages)
            )
            
            logger.debug(f"Stored chat message for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve chat history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries or None if not found
        """
        try:
            key = REDIS_KEYS["chat_session"].format(session_id=session_id)
            data = self.client.get(key)
            
            if not data:
                return None
            
            messages = json.loads(data)
            
            if limit:
                messages = messages[-limit:]
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return None
    
    # Intent Storage Methods
    
    def store_intent(self, session_id: str, intent: str, confidence: float = 0.0) -> bool:
        """
        Store the detected intent for a session.
        
        Args:
            session_id: Unique session identifier
            intent: Detected intent label
            confidence: Intent classification confidence score
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            key = REDIS_KEYS["intent"].format(session_id=session_id)
            intent_data = {
                "intent": intent,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            self.client.setex(
                key,
                REDIS_TTL["intent"],
                json.dumps(intent_data)
            )
            
            logger.debug(f"Stored intent '{intent}' for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store intent: {e}")
            return False
    
    def get_intent(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the last detected intent for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Intent data dictionary or None if not found
        """
        try:
            key = REDIS_KEYS["intent"].format(session_id=session_id)
            data = self.client.get(key)
            
            if not data:
                return None
            
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to get intent: {e}")
            return None
    
    # Session Metadata Methods
    
    def store_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Store session metadata (user_id, start_time, etc.).
        
        Args:
            session_id: Unique session identifier
            metadata: Metadata dictionary
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            key = REDIS_KEYS["session_meta"].format(session_id=session_id)
            
            # Add timestamp if not present
            if 'created_at' not in metadata:
                metadata['created_at'] = datetime.now().isoformat()
            
            metadata['last_updated'] = datetime.now().isoformat()
            
            self.client.setex(
                key,
                REDIS_TTL["session_meta"],
                json.dumps(metadata)
            )
            
            logger.debug(f"Stored metadata for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store session metadata: {e}")
            return False
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session metadata.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            key = REDIS_KEYS["session_meta"].format(session_id=session_id)
            data = self.client.get(key)
            
            if not data:
                return None
            
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            return None
    
    # Caching Methods
    
    def cache_summary(self, intent: str, summary: str, ttl: Optional[int] = None) -> bool:
        """
        Cache a summary for an intent to avoid regenerating similar responses.
        
        Args:
            intent: Intent label
            summary: Generated summary text
            ttl: Optional time-to-live override
            
        Returns:
            bool: True if cached successfully, False otherwise
        """
        try:
            key = REDIS_KEYS["summary_cache"].format(intent=intent)
            cache_data = {
                "summary": summary,
                "cached_at": datetime.now().isoformat(),
                "access_count": 1
            }
            
            ttl_value = ttl or REDIS_TTL["summary_cache"]
            self.client.setex(key, ttl_value, json.dumps(cache_data))
            
            logger.debug(f"Cached summary for intent '{intent}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache summary: {e}")
            return False
    
    def get_cached_summary(self, intent: str) -> Optional[str]:
        """
        Retrieve cached summary for an intent.
        
        Args:
            intent: Intent label
            
        Returns:
            Cached summary text or None if not found
        """
        try:
            key = REDIS_KEYS["summary_cache"].format(intent=intent)
            data = self.client.get(key)
            
            if not data:
                return None
            
            cache_data = json.loads(data)
            
            # Update access count
            cache_data["access_count"] = cache_data.get("access_count", 0) + 1
            cache_data["last_accessed"] = datetime.now().isoformat()
            
            # Update cache with new access info
            self.client.setex(
                key,
                REDIS_TTL["summary_cache"],
                json.dumps(cache_data)
            )
            
            logger.debug(f"Retrieved cached summary for intent '{intent}'")
            return cache_data["summary"]
            
        except Exception as e:
            logger.error(f"Failed to get cached summary: {e}")
            return None
    
    def cache_docs_retrieval(self, query: str, docs: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """
        Cache document retrieval results to avoid expensive ChromaDB queries.
        
        Args:
            query: Original query string
            docs: Retrieved documents list
            ttl: Optional time-to-live override
            
        Returns:
            bool: True if cached successfully, False otherwise
        """
        try:
            query_hash = self._hash_query(query)
            key = REDIS_KEYS["docs_cache"].format(query_hash=query_hash)
            
            # Generate embedding for semantic similarity
            query_embedding = self._get_query_embedding(query)
            
            cache_data = {
                "original_query": query,
                "documents": docs,
                "query_embedding": query_embedding,
                "cached_at": datetime.now().isoformat(),
                "access_count": 1
            }
            
            ttl_value = ttl or REDIS_TTL["docs_cache"]
            self.client.setex(key, ttl_value, json.dumps(cache_data))
            
            logger.debug(f"Cached {len(docs)} docs for query hash {query_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache docs retrieval: {e}")
            return False
    
    def get_cached_docs(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached document retrieval results using semantic similarity.
        
        Args:
            query: Query string to find similar cached results for
            
        Returns:
            List of cached documents or None if not found
        """
        try:
            # First try exact hash match for performance
            query_hash = self._hash_query(query)
            key = REDIS_KEYS["docs_cache"].format(query_hash=query_hash)
            data = self.client.get(key)
            
            if data:
                cache_data = json.loads(data)
                # Update access count for exact match
                cache_data["access_count"] = cache_data.get("access_count", 0) + 1
                cache_data["last_accessed"] = datetime.now().isoformat()
                
                self.client.setex(key, REDIS_TTL["docs_cache"], json.dumps(cache_data))
                
                # Handle both old and new cache format
                docs = cache_data.get("documents") or cache_data.get("docs", [])
                logger.debug(f"Retrieved {len(docs)} cached docs for exact query match")
                return docs
            
            # If no exact match, try semantic similarity
            similar_result = self.find_similar_cached_query(query)
            if similar_result:
                matched_query, cached_docs, similarity_score = similar_result
                logger.info(f"Using semantically similar cached query (similarity: {similarity_score:.3f})")
                return cached_docs
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached docs: {e}")
            return None
    
    # Lock Management Methods
    
    def acquire_lock(self, query: str, timeout: int = 300) -> bool:
        """
        Acquire a distributed lock to prevent duplicate concurrent retrievals.
        
        Args:
            query: Query string to lock on
            timeout: Lock timeout in seconds
            
        Returns:
            bool: True if lock acquired, False otherwise
        """
        try:
            query_hash = self._hash_query(query)
            key = REDIS_KEYS["locks"].format(query_hash=query_hash)
            
            # Try to set lock with NX (not exists) and EX (expiry)
            result = self.client.set(
                key,
                datetime.now().isoformat(),
                nx=True,
                ex=timeout
            )
            
            if result:
                logger.debug(f"Acquired lock for query hash {query_hash}")
                return True
            else:
                logger.debug(f"Failed to acquire lock for query hash {query_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False
    
    def release_lock(self, query: str) -> bool:
        """
        Release a distributed lock.
        
        Args:
            query: Query string to release lock for
            
        Returns:
            bool: True if lock released, False otherwise
        """
        try:
            query_hash = self._hash_query(query)
            key = REDIS_KEYS["locks"].format(query_hash=query_hash)
            
            result = self.client.delete(key)
            logger.debug(f"Released lock for query hash {query_hash}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False
    
    # Semantic Similarity Methods
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a query using SentenceTransformers model.
        
        Args:
            query: Query string to embed
            
        Returns:
            List of embedding values or None if failed
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            embedding = embeddings.embed_query(query)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def find_similar_cached_query(self, query: str) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """
        Find semantically similar cached queries using cosine similarity.
        
        Args:
            query: New query to find similar cached results for
            
        Returns:
            Tuple of (matched_query, cached_docs, similarity_score) or None
        """
        try:
            # Get embedding for new query
            query_embedding = self._get_query_embedding(query)
            if not query_embedding:
                return None
            
            # Get similarity threshold from config
            similarity_threshold = ORCHESTRATOR_CONFIG.get("cache_similarity_threshold", 0.8)
            
            # Search through cached queries
            pattern = REDIS_KEYS["docs_cache"].format(query_hash="*")
            cached_keys = self.client.keys(pattern)
            
            best_match = None
            best_score = 0.0
            
            for key in cached_keys:
                try:
                    cached_data = self.client.get(key)
                    if not cached_data:
                        continue
                        
                    cache_info = json.loads(cached_data)
                    cached_embedding = cache_info.get("query_embedding")
                    
                    if not cached_embedding:
                        continue
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > similarity_threshold and similarity > best_score:
                        best_score = similarity
                        best_match = (
                            cache_info.get("original_query", ""),
                            cache_info.get("documents", []),
                            similarity
                        )
                        
                except Exception as e:
                    logger.warning(f"Error processing cached key {key}: {e}")
                    continue
            
            if best_match:
                logger.info(f"Found similar cached query with similarity {best_score:.3f}")
                
            return best_match
            
        except Exception as e:
            logger.error(f"Failed to find similar cached query: {e}")
            return None
    
    # Utility Methods
    
    def _hash_query(self, query: str) -> str:
        """
        Generate a hash for a query string for caching purposes.
        
        Args:
            query: Query string to hash
            
        Returns:
            str: SHA256 hash of the query
        """
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all data for a specific session.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            keys_to_delete = [
                REDIS_KEYS["chat_session"].format(session_id=session_id),
                REDIS_KEYS["intent"].format(session_id=session_id),
                REDIS_KEYS["session_meta"].format(session_id=session_id),
            ]
            
            deleted_count = self.client.delete(*keys_to_delete)
            logger.info(f"Cleared {deleted_count} keys for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and health information.
        
        Returns:
            Dict containing cache statistics
        """
        try:
            info = self.client.info()
            
            stats = {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
            
            # Calculate hit rate
            hits = stats.get("keyspace_hits", 0)
            misses = stats.get("keyspace_misses", 0)
            total = hits + misses
            
            if total > 0:
                stats["hit_rate"] = hits / total
            else:
                stats["hit_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def close(self) -> None:
        """
        Close the Redis connection.
        """
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")