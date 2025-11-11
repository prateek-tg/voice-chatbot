#!/usr/bin/env python3
"""
Simplified AI Chatbot with LLM-based Semantic Caching
Architecture: Query → Cache Check (LLM semantic matching) → RAG Pipeline → Response
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from simple_redis import SimpleRedisManager
from main import RAGOrchestrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleChatbot:
    """Simplified chatbot with LLM-based semantic caching."""
    
    def __init__(self):
        """Initialize the simplified chatbot."""
        print("🚀 Initializing Simple Chatbot...")
        
        # Initialize components
        self.redis_manager = SimpleRedisManager()
        self.rag_orchestrator = RAGOrchestrator()
        self.session_id = None
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        print("✅ Simple Chatbot initialized successfully!")
    
    def create_session(self) -> str:
        """Create a new chat session."""
        self.session_id = self.redis_manager.create_session()
        logger.info(f"Created new session: {self.session_id}")
        return self.session_id
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query with cache-first approach.
        
        Args:
            user_query: User's input query
            
        Returns:
            Response dictionary with metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Step 1: Check cache first (before any other processing)
            print("🔍 Checking cache for semantic matches...")
            cached_response = self.redis_manager.find_cached_response(self.session_id, user_query)
            
            if cached_response:
                # Cache hit - return immediately
                self.cache_hits += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "response": cached_response,
                    "source": "cache",
                    "cache_hit": True,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"✅ Cache hit! Response time: {processing_time:.2f}s")
                return result
            
            # Step 2: Cache miss - proceed with RAG pipeline
            print("❌ Cache miss - processing through RAG pipeline...")
            self.cache_misses += 1
            
            # Use the existing RAG orchestrator
            rag_response = self.rag_orchestrator.process_query(
                user_query=user_query,
                session_id=self.session_id
            )
            
            if rag_response and rag_response.get("response"):
                assistant_response = rag_response["response"]
                
                # Store in cache for future use
                self.redis_manager.store_query_response(
                    self.session_id, 
                    user_query, 
                    assistant_response
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "response": assistant_response,
                    "source": "rag",
                    "cache_hit": False,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "rag_metadata": rag_response.get("retrieval", {})
                }
                
                print(f"✅ RAG response generated. Response time: {processing_time:.2f}s")
                return result
            
            else:
                # Fallback response
                fallback_response = "I apologize, but I couldn't find relevant information for your question. Could you please rephrase or ask something else?"
                
                # Still store the fallback to avoid repeated failures
                self.redis_manager.store_query_response(
                    self.session_id,
                    user_query,
                    fallback_response
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "response": fallback_response,
                    "source": "fallback",
                    "cache_hit": False,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"⚠️ Fallback response used. Response time: {processing_time:.2f}s")
                return result
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = "I encountered an error processing your request. Please try again."
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": error_response,
                "source": "error",
                "cache_hit": False,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queries": total_queries,
            "hit_rate": round(hit_rate, 1)
        }
    
    def start_interactive_chat(self) -> None:
        """Start interactive chat session."""
        print("\n" + "="*60)
        print("💬 TechGropse Simple AI Assistant")
        print("="*60)
        print("📝 Ask me anything about TechGropse!")
        print("🧠 I use smart caching with LLM semantic matching")
        print("🛑 Type 'exit', 'quit', 'bye', 'thanks', or 'that's all' to end the conversation")
        print("="*60)
        
        # Create session
        session_id = self.create_session()
        print(f"📱 Session ID: {session_id}")
        
        conversation_count = 0
        
        try:
            while True:
                conversation_count += 1
                print(f"\n💬 Question {conversation_count}")
                print("-" * 40)
                
                # Get user input
                try:
                    user_query = input("❓ You: ").strip()
                except KeyboardInterrupt:
                    break
                
                if not user_query:
                    print("🤔 Please enter a question.")
                    conversation_count -= 1
                    continue
                
                # Check for exit commands
                exit_phrases = ['exit', 'quit', 'bye', 'goodbye', 'thanks', 'thank you', 'that\'s all', 'done', 'finished']
                if user_query.lower() in exit_phrases or any(phrase in user_query.lower() for phrase in ['that\'s all', 'i\'m done']):
                    print("🤖 Assistant: Thank you for using TechGropse AI Assistant! It was great helping you today. Goodbye!")
                    break
                
                # Process query
                response_data = self.process_query(user_query)
                
                # Display response
                print(f"🤖 Assistant: {response_data['response']}")
                
                # Display metadata
                source_icon = "💾" if response_data['cache_hit'] else "🔍"
                print(f"\n{source_icon} Source: {response_data['source']} | "
                      f"⏱️ Time: {response_data['processing_time']:.2f}s")
                
                # Show cache stats periodically
                if conversation_count % 5 == 0:
                    stats = self.get_cache_stats()
                    print(f"📊 Cache Stats: {stats['hit_rate']}% hit rate "
                          f"({stats['cache_hits']}/{stats['total_queries']} hits)")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Chat interrupted by user.")
        
        # Final statistics
        print("\n" + "="*60)
        print("📊 Final Session Statistics")
        print("="*60)
        
        stats = self.get_cache_stats()
        session_stats = self.redis_manager.get_session_stats(self.session_id)
        
        print(f"💬 Total Conversations: {conversation_count - 1}")
        print(f"📝 Queries Processed: {session_stats['query_count']}")
        print(f"💾 Cache Hits: {stats['cache_hits']}")
        print(f"🔍 Cache Misses: {stats['cache_misses']}")
        print(f"📈 Cache Hit Rate: {stats['hit_rate']}%")
        print(f"⏱️ Session Duration: {session_stats.get('created_at', 'Unknown')}")
        
        print("\n👋 Chat session ended. Goodbye!")

def main():
    """Main function to start the simple chatbot."""
    try:
        chatbot = SimpleChatbot()
        chatbot.start_interactive_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    main()