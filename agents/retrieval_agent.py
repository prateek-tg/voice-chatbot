"""
Retrieval Agent for the AI Agentic RAG system.
Uses CrewAI agent interface with LangChain tools for RAG with ChromaDB and Redis caching.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
# Simplified approach - direct implementation without CrewAI tools

from config import LLM_CONFIG, RETRIEVAL_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY
from utils.redis_manager import RedisManager
from vectorstore.chromadb_client import ChromaDBClient

logger = logging.getLogger(__name__)

class PolicyRetrievalTool:
    """
    CrewAI Agent for document retrieval and RAG using LangChain tools.
    
    This agent:
    - Performs semantic search on ChromaDB
    - Checks Redis cache for previous retrievals
    - Generates summaries and responses based on retrieved documents
    - Caches results in Redis for future use
    - Uses distributed locks to prevent duplicate operations
    """
    
    def __init__(
        self, 
        chromadb_client: ChromaDBClient,
        redis_manager: RedisManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Retrieval Agent.
        
        Args:
            chromadb_client: ChromaDB client instance
            redis_manager: Redis manager instance
            config: Optional LLM configuration dictionary
        """
        self.config = config or LLM_CONFIG
        self.retrieval_config = RETRIEVAL_CONFIG
        self.chromadb_client = chromadb_client
        self.redis_manager = redis_manager
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize CrewAI agent
        self.agent = self._create_agent()
        
        # Create RAG prompt templates
        self.rag_prompt = self._create_rag_prompt()
        self.summary_prompt = self._create_summary_prompt()
    
    def _initialize_llm(self):
        """
        Initialize the LLM based on configuration.
        
        Returns:
            LangChain LLM instance
        """
        provider = self.config.get("provider", "openai")
        model = self.config.get("model", "gpt-4")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1000)
        timeout = self.config.get("timeout", 30)
        
        try:
            if provider == "demo":
                # Use demo LLM for testing without API keys
                from utils.demo_llm import DemoLLM
                return DemoLLM(
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            elif provider == "openai":
                if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-"):
                    logger.warning("OpenAI API key not configured, falling back to demo mode")
                    from utils.demo_llm import DemoLLM
                    return DemoLLM(temperature=temperature, max_tokens=max_tokens)
                
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    openai_api_key=OPENAI_API_KEY
                )
            
            elif provider == "anthropic":
                if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("your-"):
                    logger.warning("Anthropic API key not configured, falling back to demo mode")
                    from utils.demo_llm import DemoLLM
                    return DemoLLM(temperature=temperature, max_tokens=max_tokens)
                
                return ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    anthropic_api_key=ANTHROPIC_API_KEY
                )
            
            else:
                logger.warning(f"Unknown LLM provider: {provider}, using demo mode")
                from utils.demo_llm import DemoLLM
                return DemoLLM(temperature=temperature, max_tokens=max_tokens)
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}, falling back to demo mode")
            from utils.demo_llm import DemoLLM
            return DemoLLM(temperature=temperature, max_tokens=max_tokens)
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI Agent for document retrieval.
        
        Returns:
            CrewAI Agent instance
        """
        return Agent(
            role="Document Retrieval Specialist",
            goal="Retrieve relevant documents and generate accurate responses based on the knowledge base",
            backstory="""You are an expert document retrieval specialist with deep knowledge 
            of privacy policies, data protection, and company information. You excel at finding 
            the most relevant information from the knowledge base and synthesizing it into 
            clear, accurate responses. You understand the importance of caching and efficiency 
            in information retrieval.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    

    
    def _create_rag_prompt(self) -> PromptTemplate:
        """
        Create the RAG prompt template for generating responses.
        
        Returns:
            PromptTemplate for RAG responses
        """
        template = """You are a helpful assistant that answers questions based on the provided context documents and conversation history.

Context Documents:
{context}

Conversation History:
{chat_history}

Current User Question: {question}

User Intent: {intent}

Instructions:
1. FIRST, check if the question is about the conversation itself (e.g., "what was my last question?", "what did we talk about?", "can you summarize our conversation?")
   - If so, refer to the conversation history to answer, not the context documents
   - Be specific about what was discussed, questions asked, etc.

2. For questions about the documents/policies:
   - Use only the information provided in the context documents to answer the question
   - If the context doesn't contain enough information, say so
   - Be accurate and specific in your response
   - Cite relevant sections when appropriate

3. Use conversation history for context and continuity:
   - Reference previous questions or topics if relevant
   - Maintain conversational flow
   - Clarify if the current question relates to previous discussions

4. If the question is about contact information, provide complete details
5. For policy-related questions, explain clearly and mention any important conditions

Please provide a comprehensive and helpful answer that considers both the context documents and conversation history."""
        
        return PromptTemplate(
            input_variables=["context", "question", "intent", "chat_history"],
            template=template
        )
    
    def _create_summary_prompt(self) -> PromptTemplate:
        """
        Create the summary prompt template for caching.
        
        Returns:
            PromptTemplate for summaries
        """
        template = """Create a concise summary of the key information from these documents that would be useful for answering similar questions about {topic}.

Documents:
{documents}

Summary should include:
- Main points and key facts
- Important details and numbers
- Contact information if present
- Policy statements and conditions
- Any warnings or important notes

Provide a well-structured summary that captures the essential information."""
        
        return PromptTemplate(
            input_variables=["documents", "topic"],
            template=template
        )
    
    def retrieve_and_respond(
        self, 
        query: str, 
        intent: str = "general_inquiry",
        session_id: Optional[str] = None,
        chat_history: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to retrieve documents and generate a response.
        
        Args:
            query: User query string
            intent: Detected intent
            session_id: Optional session identifier
            chat_history: Formatted chat history for context
            use_cache: Whether to use Redis caching
            
        Returns:
            Dictionary with retrieval results and response
        """
        try:
            start_time = datetime.now()
            query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
            
            # Check cache first if enabled
            if use_cache:
                cached_result = self._check_cache(query, intent)
                if cached_result:
                    logger.info(f"Retrieved cached result for query: {query[:50]}...")
                    return self._format_cached_response(cached_result, query, intent, session_id, chat_history)
            
            # Acquire lock to prevent duplicate retrievals
            if use_cache and not self.redis_manager.acquire_lock(query):
                # Another process is handling this query, wait and check cache again
                import time
                time.sleep(1)
                cached_result = self._check_cache(query, intent)
                if cached_result:
                    return self._format_cached_response(cached_result, query, intent, session_id, chat_history)
            
            try:
                # Perform document retrieval
                retrieved_docs = self._retrieve_documents(query)
                
                if not retrieved_docs:
                    logger.warning(f"No documents retrieved for query: {query}")
                    return self._create_no_results_response(query, intent, session_id)
                
                # Generate response based on retrieved documents
                response = self._generate_response(query, intent, retrieved_docs, chat_history)
                
                # Create summary for caching
                summary = self._create_summary(retrieved_docs, intent)
                
                # Cache results if enabled
                if use_cache:
                    self._cache_results(query, intent, retrieved_docs, response, summary)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "query": query,
                    "intent": intent,
                    "session_id": session_id,
                    "response": response,
                    "retrieved_docs": len(retrieved_docs),
                    "documents": [self._format_document(doc) for doc in retrieved_docs],
                    "summary": summary,
                    "processing_time_seconds": processing_time,
                    "cache_hit": False,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Generated response for query: {query[:50]}... (time: {processing_time:.2f}s)")
                return result
                
            finally:
                # Always release the lock
                if use_cache:
                    self.redis_manager.release_lock(query)
                    
        except Exception as e:
            logger.error(f"Failed to retrieve and respond: {e}")
            return self._create_error_response(query, intent, session_id, str(e))
    
    def _check_cache(self, query: str, intent: str) -> Optional[Dict[str, Any]]:
        """
        Check Redis cache for previous results.
        
        Args:
            query: User query
            intent: Detected intent
            
        Returns:
            Cached result dictionary or None
        """
        # Check document cache
        cached_docs = self.redis_manager.get_cached_docs(query)
        if cached_docs:
            return {"type": "docs", "data": cached_docs}
        
        # Check summary cache
        cached_summary = self.redis_manager.get_cached_summary(intent)
        if cached_summary:
            return {"type": "summary", "data": cached_summary}
        
        return None
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents from ChromaDB.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Perform similarity search with scores
            docs_with_scores = self.chromadb_client.similarity_search_with_score(
                query=query,
                k=self.retrieval_config["top_k"]
            )
            
            # Filter by similarity threshold
            threshold = self.retrieval_config.get("similarity_threshold", 0.0)
            filtered_docs = [
                doc for doc, score in docs_with_scores 
                if score >= threshold
            ]
            
            logger.debug(f"Retrieved {len(filtered_docs)} documents for query")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _generate_response(self, query: str, intent: str, documents: List[Document], chat_history: Optional[str] = None) -> str:
        """
        Generate a response based on retrieved documents and chat history.
        
        Args:
            query: User query
            intent: Detected intent
            documents: Retrieved documents
            chat_history: Formatted chat history for context
            
        Returns:
            Generated response string
        """
        try:
            # Prepare context from documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(documents)
            ])
            
            # Format the prompt
            formatted_prompt = self.rag_prompt.format(
                context=context,
                question=query,
                intent=intent,
                chat_history=chat_history or "No previous conversation history."
            )
            
            # Generate response using LLM
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided context."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _create_summary(self, documents: List[Document], intent: str) -> str:
        """
        Create a summary of retrieved documents for caching.
        
        Args:
            documents: Retrieved documents
            intent: Detected intent
            
        Returns:
            Summary string
        """
        try:
            # Prepare documents text
            docs_text = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content[:500]}..." 
                for i, doc in enumerate(documents[:3])  # Limit for summary
            ])
            
            # Format the summary prompt
            formatted_prompt = self.summary_prompt.format(
                documents=docs_text,
                topic=intent.replace("_", " ")
            )
            
            # Generate summary using LLM
            messages = [
                SystemMessage(content="You are an expert at creating concise summaries."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return "Summary generation failed."
    
    def _cache_results(
        self, 
        query: str, 
        intent: str, 
        documents: List[Document], 
        response: str, 
        summary: str
    ) -> None:
        """
        Cache retrieval results in Redis.
        
        Args:
            query: User query
            intent: Detected intent
            documents: Retrieved documents
            response: Generated response
            summary: Created summary
        """
        try:
            # Cache documents
            docs_data = [self._format_document(doc) for doc in documents]
            self.redis_manager.cache_docs_retrieval(query, docs_data)
            
            # Cache summary by intent
            self.redis_manager.cache_summary(intent, summary)
            
            logger.debug(f"Cached results for query and intent: {intent}")
            
        except Exception as e:
            logger.error(f"Failed to cache results: {e}")
    
    def _format_document(self, document: Document) -> Dict[str, Any]:
        """
        Format a Document object for serialization.
        
        Args:
            document: LangChain Document object
            
        Returns:
            Dictionary representation of the document
        """
        return {
            "content": document.page_content,
            "metadata": document.metadata,
            "source": document.metadata.get("source", "unknown"),
            "chunk_index": document.metadata.get("chunk_index", 0)
        }
    
    def _format_cached_response(
        self, 
        cached_result: Dict[str, Any], 
        query: str, 
        intent: str, 
        session_id: Optional[str],
        chat_history: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a cached result into a standard response.
        
        Args:
            cached_result: Cached result data
            query: User query
            intent: Detected intent
            session_id: Session identifier
            
        Returns:
            Formatted response dictionary
        """
        if cached_result["type"] == "docs":
            docs_data = cached_result["data"]
            response = self._generate_response_from_cached_docs(query, intent, docs_data, chat_history)
            
            return {
                "query": query,
                "intent": intent,
                "session_id": session_id,
                "response": response,
                "retrieved_docs": len(docs_data),
                "documents": docs_data,
                "summary": "Generated from cached documents",
                "processing_time_seconds": 0.1,
                "cache_hit": True,
                "cache_type": "documents",
                "timestamp": datetime.now().isoformat()
            }
        
        elif cached_result["type"] == "summary":
            summary = cached_result["data"]
            
            return {
                "query": query,
                "intent": intent,
                "session_id": session_id,
                "response": summary,
                "retrieved_docs": 0,
                "documents": [],
                "summary": summary,
                "processing_time_seconds": 0.05,
                "cache_hit": True,
                "cache_type": "summary",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_response_from_cached_docs(self, query: str, intent: str, docs_data: List[Dict[str, Any]], chat_history: Optional[str] = None) -> str:
        """
        Generate response from cached document data.
        
        Args:
            query: User query
            intent: Detected intent
            docs_data: Cached document data
            chat_history: Previous conversation history for context
            
        Returns:
            Generated response string
        """
        try:
            # Reconstruct context from cached docs
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['content']}" 
                for i, doc in enumerate(docs_data)
            ])
            
            # Format the prompt
            formatted_prompt = self.rag_prompt.format(
                context=context,
                question=query,
                intent=intent,
                chat_history=chat_history or "No previous conversation history."
            )
            
            # Generate response using LLM
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided context."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response from cached docs: {e}")
            return "I found relevant information in my cache, but encountered an error processing it."
    
    def _create_no_results_response(self, query: str, intent: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Create response when no documents are retrieved.
        
        Args:
            query: User query
            intent: Detected intent
            session_id: Session identifier
            
        Returns:
            No results response dictionary
        """
        return {
            "query": query,
            "intent": intent,
            "session_id": session_id,
            "response": "I couldn't find specific information related to your question in the available documents. Could you please rephrase your question or provide more details?",
            "retrieved_docs": 0,
            "documents": [],
            "summary": "No relevant documents found",
            "processing_time_seconds": 0.1,
            "cache_hit": False,
            "error": "No documents retrieved",
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_response(self, query: str, intent: str, session_id: Optional[str], error: str) -> Dict[str, Any]:
        """
        Create error response when retrieval fails.
        
        Args:
            query: User query
            intent: Detected intent
            session_id: Session identifier
            error: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            "query": query,
            "intent": intent,
            "session_id": session_id,
            "response": "I apologize, but I encountered an error while processing your request. Please try again later.",
            "retrieved_docs": 0,
            "documents": [],
            "summary": "Error occurred during retrieval",
            "processing_time_seconds": 0.0,
            "cache_hit": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    # Tool methods for CrewAI
    
    def _search_documents(self, query: str) -> str:
        """
        Tool method for document search.
        
        Args:
            query: Search query
            
        Returns:
            Search results as string
        """
        try:
            documents = self._retrieve_documents(query)
            if not documents:
                return "No relevant documents found."
            
            results = []
            for i, doc in enumerate(documents):
                results.append(f"Document {i+1}: {doc.page_content[:200]}...")
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def _get_cached_retrieval(self, query: str) -> str:
        """
        Tool method for checking cached retrievals.
        
        Args:
            query: Search query
            
        Returns:
            Cached results as string or "No cache found"
        """
        try:
            cached_docs = self.redis_manager.get_cached_docs(query)
            if cached_docs:
                return f"Found {len(cached_docs)} cached documents for similar query."
            
            return "No cached results found."
            
        except Exception as e:
            return f"Error checking cache: {str(e)}"
    
    def _generate_summary(self, documents_text: str) -> str:
        """
        Tool method for generating summaries.
        
        Args:
            documents_text: Text content of documents
            
        Returns:
            Generated summary
        """
        try:
            # Simple summary generation
            messages = [
                SystemMessage(content="Create a concise summary of the key points."),
                HumanMessage(content=f"Summarize this content:\n\n{documents_text}")
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the retrieval agent configuration.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "agent_type": "Document Retrieval Specialist",
            "llm_provider": self.config.get("provider"),
            "llm_model": self.config.get("model"),
            "retrieval_config": self.retrieval_config,
            "tools": [],  # Simplified for demo
            "cache_enabled": self.redis_manager.is_connected(),
            "vectorstore_collection": self.chromadb_client.collection_name,
            "agent_role": self.agent.role,
            "agent_goal": self.agent.goal
        }


class RetrievalAgent:
    """
    Main Retrieval Agent class that wraps the PolicyRetrievalTool for easier use.
    """
    
    def __init__(self, chromadb_client: ChromaDBClient, redis_manager: RedisManager):
        """
        Initialize the Retrieval Agent.
        
        Args:
            chromadb_client: ChromaDB client instance
            redis_manager: Redis manager instance
        """
        self.tool = PolicyRetrievalTool(
            chromadb_client=chromadb_client,
            redis_manager=redis_manager
        )
        self.agent = self.tool.agent
    
    def retrieve_and_respond(self, query: str, intent: str = "general_inquiry", session_id: Optional[str] = None, chat_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query and return retrieval results.
        
        Args:
            query: User query string
            intent: Detected intent
            session_id: Optional session identifier
            chat_history: Previous conversation history for context
            
        Returns:
            Dictionary with retrieval results and response
        """
        return self.tool.retrieve_and_respond(
            query=query,
            intent=intent,
            session_id=session_id,
            chat_history=chat_history
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information.
        
        Returns:
            Dictionary with agent information
        """
        return self.tool.get_agent_info()