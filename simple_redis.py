#!/usr/bin/env python3
"""
Simplified Redis Manager for Query-Response Caching
Stores only user queries and assistant responses with LLM-based semantic matching.
"""

import redis
import json
import uuid
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRedisManager:
    """Simple Redis manager for query-response pairs with LLM semantic matching."""
    
    def __init__(self):
        """Initialize Redis connection and LLM for semantic matching."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            
            # Initialize LLM for semantic matching
            self.matching_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,  # Deterministic for consistency
                max_tokens=10,    # Just need "YES" or "NO"
                timeout=5
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def create_session(self) -> str:
        """Create a new session ID."""
        session_id = str(uuid.uuid4())
        session_key = f"session:{session_id}"
        
        # Initialize session metadata
        session_data = {
            "created_at": datetime.now().isoformat(),
            "query_count": 0
        }
        
        self.redis_client.hset(session_key, mapping=session_data)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def store_query_response(self, session_id: str, user_query: str, assistant_response: str) -> None:
        """
        Store a query-response pair for a session.
        
        Args:
            session_id: Session identifier
            user_query: User's query
            assistant_response: Assistant's response
        """
        try:
            # Update session query count
            session_key = f"session:{session_id}"
            query_count = self.redis_client.hincrby(session_key, "query_count", 1)
            
            # Store query-response pair
            qa_key = f"session:{session_id}:qa:{query_count}"
            qa_data = {
                "user_query": user_query,
                "assistant_response": assistant_response,
                "timestamp": datetime.now().isoformat()
            }
            
            self.redis_client.hset(qa_key, mapping=qa_data)
            
            # Set expiration (24 hours)
            self.redis_client.expire(qa_key, 86400)
            
            logger.info(f"Stored Q&A pair {query_count} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store query-response: {e}")
    
    def get_last_queries(self, session_id: str, limit: int = 7) -> List[Dict[str, Any]]:
        """
        Get the last N queries and responses for a session.
        
        Args:
            session_id: Session identifier
            limit: Number of recent queries to retrieve
            
        Returns:
            List of query-response dictionaries
        """
        try:
            session_key = f"session:{session_id}"
            query_count = self.redis_client.hget(session_key, "query_count")
            
            if not query_count:
                return []
            
            query_count = int(query_count)
            queries = []
            
            # Get last N queries (or all if less than N)
            start_idx = max(1, query_count - limit + 1)
            
            for i in range(start_idx, query_count + 1):
                qa_key = f"session:{session_id}:qa:{i}"
                qa_data = self.redis_client.hgetall(qa_key)
                
                if qa_data:
                    queries.append({
                        "query_id": i,
                        "user_query": qa_data.get("user_query", ""),
                        "assistant_response": qa_data.get("assistant_response", ""),
                        "timestamp": qa_data.get("timestamp", "")
                    })
            
            return queries
            
        except Exception as e:
            logger.error(f"Failed to get last queries: {e}")
            return []
    
    def check_semantic_match(self, current_query: str, cached_queries: List[Dict[str, Any]]) -> Optional[str]:
        """
        Check if current query semantically matches any cached query using LLM.
        
        Args:
            current_query: The current user query
            cached_queries: List of cached query-response pairs
            
        Returns:
            Assistant response if match found, None otherwise
        """
        try:
            if not cached_queries:
                return None
            
            # Create comparison prompt
            cached_queries_text = "\n".join([
                f"{i+1}. {q['user_query']}" 
                for i, q in enumerate(cached_queries)
            ])
            
            system_prompt = f"""You are comparing queries for semantic similarity and intent matching.

Current user query: "{current_query}"

Previous queries from this session:
{cached_queries_text}

Task: Determine if the current query has the EXACT SAME intent as any previous query.
- Match ONLY if asking for the same information in a different way
- DO NOT match if asking for MORE/ADDITIONAL information on the same topic
- DO NOT match if the request type is different (summary vs details vs more info)
- Focus on whether the user wants the SAME answer or DIFFERENT information

Respond with ONLY:
- "YES X" if EXACT same intent (where X is the number 1-{len(cached_queries)} of the matching query)
- "NO" if different intent or asking for additional/more information

Examples:
- "What is your privacy policy?" vs "Tell me about privacy policy" → YES (same info)
- "What is child safety?" vs "I need more data on child safety" → NO (asking for additional info)
- "How do you use cookies?" vs "Tell me about cookie usage" → YES (same info)
- "Summarize our chat" vs "Can you summarize what we discussed?" → YES (same request)
- "Hello" vs "Hi there" → YES (same greeting)"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_query)
            ]
            
            response = self.matching_llm.invoke(messages)
            result = response.content.strip().upper()
            
            # Parse response
            if result.startswith("YES"):
                try:
                    # Extract query number
                    parts = result.split()
                    if len(parts) > 1:
                        query_idx = int(parts[1]) - 1  # Convert to 0-based index
                        if 0 <= query_idx < len(cached_queries):
                            matched_response = cached_queries[query_idx]["assistant_response"]
                            logger.info(f"Semantic match found with cached query {query_idx + 1}")
                            return matched_response
                except (ValueError, IndexError):
                    pass
            
            logger.info("No semantic match found in cached queries")
            return None
            
        except Exception as e:
            logger.error(f"Failed to check semantic match: {e}")
            return None
    
    def find_cached_response(self, session_id: str, user_query: str) -> Optional[str]:
        """
        Find a cached response for the user query if semantically similar.
        
        Args:
            session_id: Session identifier
            user_query: Current user query
            
        Returns:
            Cached response if match found, None otherwise
        """
        try:
            # Get last 7 queries
            cached_queries = self.get_last_queries(session_id, 7)
            
            if not cached_queries:
                logger.info("No cached queries found for session")
                return None
            
            logger.info(f"Checking semantic similarity with {len(cached_queries)} cached queries")
            
            # Check for semantic match
            cached_response = self.check_semantic_match(user_query, cached_queries)
            
            if cached_response:
                logger.info("✅ Cache hit - returning cached response")
                return cached_response
            else:
                logger.info("❌ Cache miss - no semantic match found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to find cached response: {e}")
            return None
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.hgetall(session_key)
            
            return {
                "session_id": session_id,
                "created_at": session_data.get("created_at", ""),
                "query_count": int(session_data.get("query_count", 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"session_id": session_id, "query_count": 0}

def main():
    """Test the simple Redis manager."""
    manager = SimpleRedisManager()
    
    # Create test session
    session_id = manager.create_session()
    print(f"Created session: {session_id}")
    
    # Test queries
    test_queries = [
        ("What is your privacy policy?", "We have a comprehensive privacy policy that protects your data..."),
        ("How do you handle personal information?", "We handle personal information with strict security measures..."),
        ("Tell me about your services", "We offer mobile app development, web development, and AI solutions...")
    ]
    
    # Store test queries
    for query, response in test_queries:
        manager.store_query_response(session_id, query, response)
        print(f"Stored: {query[:50]}...")
    
    # Test semantic matching
    test_query = "What's your data privacy approach?"
    cached_response = manager.find_cached_response(session_id, test_query)
    
    if cached_response:
        print(f"\n✅ Found cached response for: {test_query}")
        print(f"Response: {cached_response[:100]}...")
    else:
        print(f"\n❌ No cached response found for: {test_query}")

if __name__ == "__main__":
    main()