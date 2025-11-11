"""
Main Orchestrator for the AI Agentic RAG system.
Coordinates Intent Agent and Retrieval Agent, manages sessions, and handles the complete RAG workflow.

ARCHITECTURE:
USER → ORCHESTRATOR → [Intent Agent + Retrieval Agent] → [Redis + ChromaDB] → OUTPUT
"""

import logging
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from utils.redis_manager import RedisManager
from vectorstore.chromadb_client import ChromaDBClient
from config import get_config, validate_config, ORCHESTRATOR_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    Central Orchestrator for the AI Agentic RAG system.
    
    This orchestrator:
    - Acts as the central controller accepting user input
    - Coordinates Intent Agent and Retrieval Agent
    - Manages session state and conversation history in Redis
    - Combines results and returns final answers
    - Handles caching and session management
    """
    
    def __init__(self):
        """
        Initialize the RAG Orchestrator with all required components.
        """
        logger.info("Initializing RAG Orchestrator...")
        
        # Load and validate configuration
        self.config = get_config()
        if not validate_config():
            logger.warning("Configuration validation failed - some features may not work properly")
        
        self.orchestrator_config = ORCHESTRATOR_CONFIG
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Orchestrator initialized successfully")
    
    def _initialize_components(self) -> None:
        """
        Initialize all system components.
        """
        try:
            # Initialize Redis Manager
            logger.info("Initializing Redis Manager...")
            self.redis_manager = RedisManager(self.config["redis"])
            
            # Initialize ChromaDB Client
            logger.info("Initializing ChromaDB Client...")
            self.chromadb_client = ChromaDBClient(self.config["chromadb"])
            
            # Initialize Intent Agent
            logger.info("Initializing Intent Agent...")
            self.intent_agent = IntentAgent(self.config["llm"])
            
            # Initialize Retrieval Agent
            logger.info("Initializing Retrieval Agent...")
            self.retrieval_agent = RetrievalAgent(
                chromadb_client=self.chromadb_client,
                redis_manager=self.redis_manager
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_query(
        self, 
        user_query: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to process a user query through the complete RAG pipeline.
        
        WORKFLOW:
        1. Create or retrieve session
        2. Store user query in Redis chat history
        3. Call Intent Agent to classify intent
        4. Store intent in Redis
        5. Check for cached responses
        6. Call Retrieval Agent if needed
        7. Combine and return final response
        8. Update session metadata
        
        Args:
            user_query: The user's question or request
            session_id: Optional session identifier (will create new if not provided)
            user_id: Optional user identifier
            
        Returns:
            Dictionary containing the complete response and metadata
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Handle session management
            if not session_id:
                session_id = self._create_new_session(user_id)
            
            # Step 2: Store user query in chat history
            user_message = {
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now().isoformat()
            }
            self.redis_manager.store_chat_message(session_id, user_message)
            
            # Step 3: Get chat history for context
            chat_history = self._get_formatted_chat_history(session_id)
            
            logger.info(f"Processing query for session {session_id}: {user_query[:100]}...")
            
            # Step 4: Check for simple greetings first (instant response)
            if self._is_simple_greeting(user_query):
                logger.info("Processing simple greeting with instant response...")
                retrieval_result = self._generate_instant_greeting_response(user_query, session_id)
                
                # Create minimal final response for greeting
                final_response = self._create_final_response(
                    user_query=user_query,
                    session_id=session_id,
                    intent_result={"intent": "simple_greeting", "confidence": 1.0, "reasoning": "Direct greeting detection"},
                    retrieval_result=retrieval_result,
                    start_time=start_time
                )
                
                # Store assistant response in chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": final_response["response"],
                    "timestamp": datetime.now().isoformat(),
                    "intent": "simple_greeting",
                    "confidence": 1.0
                }
                self.redis_manager.store_chat_message(session_id, assistant_message)
                
                return final_response
            
            # Step 5: Classify intent using Intent Agent
            logger.info("Classifying user intent...")
            intent_result = self.intent_agent.classify_intent(
                query=user_query,
                chat_history=chat_history,
                session_id=session_id
            )
            
            detected_intent = intent_result["intent"]
            intent_confidence = intent_result["confidence"]
            
            # Step 6: Store intent in Redis
            self.redis_manager.store_intent(session_id, detected_intent, intent_confidence)
            
            logger.info(f"Detected intent: {detected_intent} (confidence: {intent_confidence:.2f})")
            
            # Step 7: Check if this is a casual intent that can be handled quickly
            if self._is_casual_intent(detected_intent):
                logger.info("Processing casual intent with quick response...")
                casual_result = self._generate_casual_response(user_query, detected_intent, session_id, chat_history)
                
                # Create response in same format as retrieval result
                retrieval_result = {
                    "response": casual_result["response"],
                    "query": user_query,
                    "intent": detected_intent,
                    "session_id": session_id,
                    "retrieved_docs": 0,
                    "documents": [],
                    "summary": "Quick casual response - no document retrieval needed",
                    "processing_time_seconds": 0.1,
                    "cache_hit": False,
                    "cache_type": "none",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Step 7: Check if we should use caching
                use_cache = self.orchestrator_config.get("enable_caching", True)
                
                # Step 8: Retrieve documents and generate response using Retrieval Agent
                logger.info("Retrieving documents and generating response...")
                retrieval_result = self.retrieval_agent.retrieve_and_respond(
                    query=user_query,
                    intent=detected_intent,
                    session_id=session_id,
                    chat_history=chat_history
                )
            
            # Step 9: Combine results into final response
            final_response = self._create_final_response(
                user_query=user_query,
                session_id=session_id,
                intent_result=intent_result,
                retrieval_result=retrieval_result,
                start_time=start_time
            )
            
            # Step 10: Store assistant response in chat history
            assistant_message = {
                "role": "assistant", 
                "content": final_response["response"],
                "timestamp": datetime.now().isoformat(),
                "intent": detected_intent,
                "confidence": intent_confidence
            }
            self.redis_manager.store_chat_message(session_id, assistant_message)
            
            # Step 10: Update session metadata
            self._update_session_metadata(session_id, user_id, final_response)
            
            logger.info(f"Successfully processed query in {final_response['performance']['total_processing_time']:.2f}s")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(user_query, session_id, str(e))
    
    def _create_new_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session with metadata.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            New session ID
        """
        session_id = str(uuid.uuid4())
        
        metadata = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "query_count": 0,
            "last_activity": datetime.now().isoformat()
        }
        
        self.redis_manager.store_session_metadata(session_id, metadata)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session (public method).
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            New session ID
        """
        return self._create_new_session(user_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from Redis manager.
        
        Returns:
            Cache statistics dictionary
        """
        return self.redis_manager.get_cache_stats()
    
    def _get_formatted_chat_history(self, session_id: str, limit: Optional[int] = None) -> str:
        """
        Get formatted chat history for context.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            Formatted chat history string
        """
        limit = limit or self.orchestrator_config.get("max_chat_history", 10)
        
        messages = self.redis_manager.get_chat_history(session_id, limit)
        if not messages:
            return "No previous conversation history."
        
        formatted_history = []
        for msg in messages[-limit:]:  # Get last N messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _create_final_response(
        self,
        user_query: str,
        session_id: str,
        intent_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Create the final combined response.
        
        Args:
            user_query: Original user query
            session_id: Session identifier
            intent_result: Intent classification result  
            retrieval_result: Document retrieval result
            start_time: Processing start time
            
        Returns:
            Complete response dictionary
        """
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Get session metadata
        session_meta = self.redis_manager.get_session_metadata(session_id) or {}
        
        return {
            # Core response
            "response": retrieval_result.get("response", "I apologize, but I couldn't generate a response."),
            "session_id": session_id,
            "query": user_query,
            
            # Intent information
            "intent": {
                "label": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "reasoning": intent_result.get("reasoning", ""),
                "alternatives": intent_result.get("alternative_intents", [])
            },
            
            # Retrieval information
            "retrieval": {
                "documents_found": retrieval_result.get("retrieved_docs", 0),
                "cache_hit": retrieval_result.get("cache_hit", False),
                "cache_type": retrieval_result.get("cache_type", None),
                "processing_time": retrieval_result.get("processing_time_seconds", 0.0)
            },
            
            # Performance metrics
            "performance": {
                "total_processing_time": total_time,
                "intent_classification_time": 0.5,  # Estimated
                "retrieval_time": retrieval_result.get("processing_time_seconds", 0.0)
            },
            
            # Session information
            "session": {
                "session_id": session_id,
                "user_id": session_meta.get("user_id"),
                "query_count": session_meta.get("query_count", 0) + 1,
                "created_at": session_meta.get("created_at"),
                "last_activity": datetime.now().isoformat()
            },
            
            # System information
            "system": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "components": {
                    "redis_connected": self.redis_manager.is_connected(),
                    "chromadb_collection": self.chromadb_client.collection_name,
                    "intent_agent": "operational",
                    "retrieval_agent": "operational"
                }
            },
            
            # Debug information (can be removed in production)
            "debug": {
                "intent_raw_response": intent_result.get("raw_response", ""),
                "retrieval_summary": retrieval_result.get("summary", ""),
                "documents": retrieval_result.get("documents", [])
            }
        }
    
    def _update_session_metadata(self, session_id: str, user_id: Optional[str], response: Dict[str, Any]) -> None:
        """
        Update session metadata with latest activity.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            response: Response data for metadata update
        """
        try:
            metadata = self.redis_manager.get_session_metadata(session_id) or {}
            
            # Update metadata
            metadata.update({
                "last_activity": datetime.now().isoformat(),
                "query_count": metadata.get("query_count", 0) + 1,
                "last_intent": response["intent"]["label"],
                "last_intent_confidence": response["intent"]["confidence"],
                "total_processing_time": metadata.get("total_processing_time", 0.0) + response["performance"]["total_processing_time"]
            })
            
            if user_id:
                metadata["user_id"] = user_id
            
            self.redis_manager.store_session_metadata(session_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
    
    def _create_error_response(self, user_query: str, session_id: Optional[str], error: str) -> Dict[str, Any]:
        """
        Create an error response when processing fails.
        
        Args:
            user_query: Original user query
            session_id: Session identifier
            error: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            "response": "I apologize, but I encountered an error while processing your request. Please try again later or contact support if the issue persists.",
            "session_id": session_id,
            "query": user_query,
            "error": True,
            "error_message": error,
            "intent": {
                "label": "error",
                "confidence": 0.0,
                "reasoning": "Error occurred during processing"
            },
            "retrieval": {
                "documents_found": 0,
                "cache_hit": False,
                "processing_time": 0.0
            },
            "performance": {
                "total_processing_time": 0.0,
                "intent_classification_time": 0.0,
                "retrieval_time": 0.0
            },
            "system": {
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        }
    
    def _is_simple_greeting(self, query: str) -> bool:
        """
        Use LLM to intelligently detect if query is a simple greeting that can be handled instantly.
        
        Args:
            query: User query string
            
        Returns:
            True if query is a simple greeting
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Use fast, cheap model for quick detection
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=10
            )
            
            system_prompt = """You are a greeting detector. Analyze if the user input is a simple greeting or casual hello that requires no specific information.

Examples of simple greetings: "hi", "hello", "hey there", "good morning", "howdy", "what's up", "yo"
Examples of NOT simple greetings: "hi, what are your data policies?", "hello, I need help with cookies", "hey, tell me about privacy"

Respond with only "YES" if it's a simple greeting, or "NO" if it needs specific information or has additional questions."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User input: '{query}'")
            ]
            
            response = llm.invoke(messages)
            return response.content.strip().upper() == "YES"
            
        except Exception as e:
            logger.error(f"Failed to detect greeting with LLM: {e}")
            # Fallback to simple check for common greetings
            query_lower = query.lower().strip()
            return query_lower in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    
    def _generate_instant_greeting_response(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """
        Generate instant response for simple greetings using LLM for natural responses.
        
        Args:
            user_query: User's greeting
            session_id: Session identifier
            
        Returns:
            Instant greeting response dictionary
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Use fast model for quick greeting response
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=50
            )
            
            system_prompt = """You are a friendly AI assistant for TechGropse, a technology company. 
Respond naturally to the user's greeting and offer to help with TechGropse information. 
Keep responses brief, warm, and professional. Always offer assistance."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = llm.invoke(messages).content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate greeting response with LLM: {e}")
            # Fallback to simple response
            response = "Hello! How can I assist you with TechGropse today?"
        
        return {
            "response": response,
            "query": user_query,
            "intent": "simple_greeting",
            "session_id": session_id,
            "retrieved_docs": 0,
            "documents": [],
            "summary": "Instant greeting response - no processing needed",
            "processing_time_seconds": 0.01,
            "cache_hit": False,
            "cache_type": "none",
            "timestamp": datetime.now().isoformat()
        }
    
    def _is_casual_intent(self, intent: str) -> bool:
        """
        Check if an intent should be handled quickly without document retrieval.
        
        Args:
            intent: Detected intent string
            
        Returns:
            True if intent is casual and should be handled quickly
        """
        # Only pure greetings should be handled casually - EVERYTHING else about TechGropse should go through RAG
        pure_greeting_patterns = [
            'greeting', 'hello', 'hi', 'farewell', 'goodbye', 'bye',
            'initiat', 'greeting_initiation', 'conversation_starter'  # 'initiat' catches initiate/initiating
        ]
        
        intent_lower = intent.lower()
        
        # Check if it's a pure greeting
        is_pure_greeting = any(pattern in intent_lower for pattern in pure_greeting_patterns)
        
        # If intent contains ANY business/information-related keywords, it should go through RAG
        business_keywords = [
            'techgropse', 'company', 'policy', 'privacy', 'data', 'information', 
            'safety', 'child', 'cookie', 'contact', 'service', 'security',
            'measures', 'practices', 'about', 'what', 'how', 'details',
            'inquiry', 'seeking', 'request'
        ]
        
        has_business_keywords = any(keyword in intent_lower for keyword in business_keywords)
        
        # Only return True for pure greetings that don't have any business/information keywords
        return is_pure_greeting and not has_business_keywords
    
    def _generate_casual_response(self, user_query: str, intent: str, session_id: str, chat_history: str) -> Dict[str, Any]:
        """
        Generate a quick response for casual conversation without document retrieval.
        
        Args:
            user_query: User's query
            intent: Detected intent
            session_id: Session identifier
            chat_history: Formatted chat history
            
        Returns:
            Quick response dictionary
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        try:
            # Initialize LLM for quick responses
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use faster, cheaper model for casual conversation
                temperature=0.7,
                max_tokens=200  # Keep responses concise
            )
            
            # Create system prompt for casual conversation
            system_prompt = f"""You are a helpful AI assistant for TechGropse, a technology company. 
Handle this casual conversation naturally and briefly. Keep responses friendly, professional, and concise.

For greetings: Be warm and ask how you can help.
For personal questions: Explain you're an AI assistant for TechGropse that helps with company information.
For thanks: Acknowledge gracefully and offer further assistance.
For chit-chat: Be friendly but redirect to how you can help with TechGropse information.

Intent detected: {intent}
Chat history: {chat_history or "No previous conversation."}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = llm.invoke(messages)
            casual_response = response.content.strip()
            
            return {
                "response": casual_response,
                "intent": intent,
                "confidence": 1.0,
                "processing_type": "casual_quick_response",
                "documents_used": 0,
                "cache_hit": False
            }
            
        except Exception as e:
            logger.error(f"Failed to generate casual response: {e}")
            # Fallback responses
            fallback_responses = {
                "greeting": "Hello! I'm here to help you with information about TechGropse. How can I assist you today?",
                "personal": "I'm an AI assistant created to help with TechGropse information and services. How can I help you?",
                "thanks": "You're welcome! Is there anything else about TechGropse I can help you with?",
                "goodbye": "Thank you for using TechGropse's AI assistant. Have a great day!"
            }
            
            # Simple fallback logic
            response_key = "greeting"
            if "thank" in user_query.lower():
                response_key = "thanks"
            elif "bye" in user_query.lower() or "goodbye" in user_query.lower():
                response_key = "goodbye"
            elif any(word in user_query.lower() for word in ["who", "what", "name", "you"]):
                response_key = "personal"
            
            return {
                "response": fallback_responses[response_key],
                "intent": intent,
                "confidence": 1.0,
                "processing_type": "casual_fallback_response",
                "documents_used": 0,
                "cache_hit": False
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        try:
            # Get session metadata
            metadata = self.redis_manager.get_session_metadata(session_id)
            if not metadata:
                return {"error": "Session not found"}
            
            # Get chat history
            chat_history = self.redis_manager.get_chat_history(session_id)
            
            # Get last intent
            intent_data = self.redis_manager.get_intent(session_id)
            
            return {
                "session_id": session_id,
                "metadata": metadata,
                "message_count": len(chat_history) if chat_history else 0,
                "last_messages": chat_history[-5:] if chat_history else [],
                "last_intent": intent_data,
                "session_active": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {"error": str(e)}
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all data for a specific session.
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            result = self.redis_manager.clear_session(session_id)
            logger.info(f"Cleared session: {session_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and health information.
        
        Returns:
            System status dictionary
        """
        try:
            # Redis status
            redis_stats = self.redis_manager.get_cache_stats()
            redis_connected = self.redis_manager.is_connected()
            
            # ChromaDB status
            chromadb_info = self.chromadb_client.get_collection_info()
            
            # Agent status
            intent_agent_info = self.intent_agent.get_agent_info()
            retrieval_agent_info = self.retrieval_agent.get_agent_info()
            
            return {
                "system": {
                    "status": "operational",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0"
                },
                "redis": {
                    "connected": redis_connected,
                    "stats": redis_stats
                },
                "chromadb": {
                    "collection_info": chromadb_info
                },
                "agents": {
                    "intent_agent": intent_agent_info,
                    "retrieval_agent": retrieval_agent_info
                },
                "configuration": {
                    "caching_enabled": self.orchestrator_config.get("enable_caching", True),
                    "max_chat_history": self.orchestrator_config.get("max_chat_history", 10),
                    "session_timeout": self.orchestrator_config.get("session_timeout", 3600)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "system": {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def close(self) -> None:
        """
        Clean shutdown of all components.
        """
        try:
            logger.info("Shutting down RAG Orchestrator...")
            
            if hasattr(self, 'redis_manager'):
                self.redis_manager.close()
            
            if hasattr(self, 'chromadb_client'):
                self.chromadb_client.close()
            
            logger.info("RAG Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """
    Main function to demonstrate the RAG system.
    """
    try:
        # Initialize orchestrator
        orchestrator = RAGOrchestrator()
        
        # Get system status
        status = orchestrator.get_system_status()
        print("\n=== SYSTEM STATUS ===")
        print(json.dumps(status, indent=2))
        
        # Example queries
        example_queries = [
            "What information do you collect about users?",
            "How can I contact TechGropse?",
            "What is your policy on cookies?",
            "Do you share personal data with third parties?",
            "What security measures do you have for protecting my data?"
        ]
        
        print("\n=== EXAMPLE INTERACTIONS ===")
        
        for i, query in enumerate(example_queries):
            print(f"\n--- Query {i+1} ---")
            print(f"User: {query}")
            
            # Process query
            response = orchestrator.process_query(query)
            
            print(f"Assistant: {response['response']}")
            print(f"Intent: {response['intent']['label']} (confidence: {response['intent']['confidence']:.2f})")
            print(f"Processing time: {response['performance']['total_processing_time']:.2f}s")
            print(f"Cache hit: {response['retrieval']['cache_hit']}")
            
            # Short pause between queries
            import time
            time.sleep(1)
        
        # Show session info
        session_id = response['session_id']
        session_info = orchestrator.get_session_info(session_id)
        print(f"\n=== SESSION INFO ===")
        print(f"Session ID: {session_id}")
        print(f"Total messages: {session_info['message_count']}")
        print(f"Queries processed: {session_info['metadata']['query_count']}")
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'orchestrator' in locals():
            orchestrator.close()


if __name__ == "__main__":
    main()