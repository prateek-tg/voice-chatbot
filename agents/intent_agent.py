"""
Intent Agent for the AI Agentic RAG system.
Uses CrewAI's agent interface with LangChain LLM to classify user intents.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from crewai import Agent, Task
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from config import LLM_CONFIG, INTENT_LABELS, OPENAI_API_KEY, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

class IntentAgent:
    """
    CrewAI Agent for intent classification using LangChain LLMs.
    
    This agent:
    - Analyzes user queries to determine intent
    - Returns intent labels with confidence scores
    - Uses configurable LLM providers (OpenAI, Anthropic)
    - Provides structured JSON responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Intent Agent.
        
        Args:
            config: Optional LLM configuration dictionary
        """
        self.config = config or LLM_CONFIG
        self.intent_labels = INTENT_LABELS
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize CrewAI agent
        self.agent = self._create_agent()
        
        # Create intent classification prompt
        self.intent_prompt = self._create_intent_prompt()
    
    def _initialize_llm(self):
        """
        Initialize the LLM based on configuration.
        
        Returns:
            LangChain LLM instance
        """
        provider = self.config.get("provider", "openai")
        model = self.config.get("model", "gpt-4o-mini")
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
        Create the CrewAI Agent for flexible query analysis.
        
        Returns:
            CrewAI Agent instance
        """
        return Agent(
            role="Query Analyzer",
            goal="Deeply understand and analyze user queries to determine their true intent and information needs",
            backstory="""You are an expert query analysis agent who excels at understanding 
            the nuanced meaning behind user questions. Rather than forcing queries into 
            rigid categories, you analyze what users actually want to know and provide 
            detailed insights about their information needs. You understand that similar 
            words can have very different intents based on context and phrasing.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _create_intent_prompt(self) -> PromptTemplate:
        """
        Create the flexible intent analysis prompt template.
        
        Returns:
            PromptTemplate for intent analysis
        """
        template = """You are an expert query analyzer. Your job is to deeply understand what the user is asking for, rather than forcing their question into predefined categories.

User Query: "{query}"

Chat History (for context):
{chat_history}

Analyze this query comprehensively and provide insights about:

1. **Core Subject**: What is the main topic/subject they're asking about?
2. **Information Type**: What specific type of information do they want? (definitions, procedures, lists, contact details, policies, etc.)
3. **Query Specificity**: How specific or broad is their request?
4. **User Intent**: What do they actually want to accomplish?
5. **Context Clues**: Any important nuances, implied meanings, or context from their phrasing?

Examples of nuanced analysis:
- "What types of cookies do you use?" → Asking for categorization/classification of cookies (specific list)
- "What is your cookie policy?" → Asking for the overall policy document/summary (general policy)
- "How do you collect data?" → Asking about procedures/methods (process-focused)
- "What data do you collect?" → Asking for inventory/list of data types (content-focused)

Respond with a JSON object:
{{
    "intent": "descriptive_intent_based_on_actual_analysis",
    "confidence": confidence_score_between_0_and_1,
    "reasoning": "detailed explanation of what the user is actually asking for",
    "query_type": "classification|explanation|procedure|contact|list|comparison|etc",
    "subject_area": "the main topic area",
    "specificity_level": "very_specific|specific|general|very_general",
    "semantic_keywords": ["key", "concepts", "from", "query"],
    "expected_response_type": "what type of response would best answer this query",
    "timestamp": "{timestamp}"
}}

DO NOT force the query into predefined categories. Instead, analyze what the user is genuinely asking for and create a descriptive intent that captures their actual need."""
        
        return PromptTemplate(
            input_variables=["query", "chat_history", "timestamp"],
            template=template
        )
    
    def classify_intent(
        self, 
        query: str, 
        chat_history: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify the intent of a user query.
        
        Args:
            query: User query string
            chat_history: Optional chat history for context
            session_id: Optional session identifier
            
        Returns:
            Dictionary with intent classification results
        """
        try:
            # Prepare chat history context
            history_context = chat_history or "No previous chat history available."
            timestamp = datetime.now().isoformat()
            
            # Format the prompt
            formatted_prompt = self.intent_prompt.format(
                query=query,
                chat_history=history_context,
                timestamp=timestamp
            )
            
            # Create task for the agent
            task = Task(
                description=f"Classify the intent of this user query: {query}",
                agent=self.agent,
                expected_output="JSON object with intent classification results"
            )
            
            # Execute the task
            messages = [
                SystemMessage(content="You are an expert intent classifier."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response
            result = self._parse_intent_response(response.content, query, session_id)
            
            logger.info(f"Classified intent for query '{query[:50]}...': {result['intent']} (confidence: {result['confidence']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to classify intent: {e}")
            return self._create_fallback_result(query, session_id, str(e))
    
    def _parse_intent_response(self, response: str, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured intent result.
        
        Args:
            response: Raw LLM response
            query: Original user query
            session_id: Optional session identifier
            
        Returns:
            Structured intent classification result
        """
        try:
            import json
            
            # Try to extract JSON from response
            response_clean = response.strip()
            
            # Handle potential markdown formatting
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            
            # Parse JSON
            parsed_result = json.loads(response_clean)
            
            # Validate and clean result with new flexible format
            result = {
                "intent": parsed_result.get("intent", "general_inquiry"),
                "confidence": float(parsed_result.get("confidence", 0.5)),
                "reasoning": parsed_result.get("reasoning", "No reasoning provided"),
                "query_type": parsed_result.get("query_type", "general"),
                "subject_area": parsed_result.get("subject_area", "unknown"),
                "specificity_level": parsed_result.get("specificity_level", "general"),
                "semantic_keywords": parsed_result.get("semantic_keywords", []),
                "expected_response_type": parsed_result.get("expected_response_type", "explanation"),
                "timestamp": parsed_result.get("timestamp", datetime.now().isoformat()),
                "query": query,
                "session_id": session_id,
                "raw_response": response
            }
            
            # No more rigid validation - allow flexible intents
            # The system will use the intent for caching and routing but won't restrict it
            
            # Ensure confidence is in valid range
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._extract_intent_from_text(response, query, session_id)
        
        except Exception as e:
            logger.error(f"Error parsing intent response: {e}")
            return self._create_fallback_result(query, session_id, str(e))
    
    def _extract_intent_from_text(self, response: str, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Extract intent from text response when JSON parsing fails.
        
        Args:
            response: Raw LLM response
            query: Original user query  
            session_id: Optional session identifier
            
        Returns:
            Best-effort intent classification result
        """
        # Look for intent labels in the response
        response_lower = response.lower()
        found_intents = []
        
        for intent in self.intent_labels:
            if intent.lower() in response_lower:
                found_intents.append(intent)
        
        # Use the first found intent or default
        detected_intent = found_intents[0] if found_intents else "general_inquiry"
        
        # Estimate confidence based on response content
        confidence = 0.7 if found_intents else 0.3
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "reasoning": f"Extracted from text response: {response[:100]}...",
            "alternative_intents": found_intents[1:3] if len(found_intents) > 1 else [],
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "session_id": session_id,
            "raw_response": response,
            "parsing_error": "Failed to parse JSON response"
        }
    
    def _create_fallback_result(self, query: str, session_id: Optional[str], error: str) -> Dict[str, Any]:
        """
        Create a fallback intent result when classification fails.
        
        Args:
            query: Original user query
            session_id: Optional session identifier
            error: Error message
            
        Returns:
            Fallback intent classification result
        """
        return {
            "intent": "general_inquiry",
            "confidence": 0.1,
            "reasoning": f"Fallback due to error: {error}",
            "alternative_intents": [],
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "session_id": session_id,
            "raw_response": "",
            "error": error
        }
    
    def batch_classify_intents(self, queries: list[str], session_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """
        Classify intents for multiple queries in batch.
        
        Args:
            queries: List of user queries
            session_id: Optional session identifier
            
        Returns:
            List of intent classification results
        """
        results = []
        
        for query in queries:
            try:
                result = self.classify_intent(query, session_id=session_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify query '{query}': {e}")
                results.append(self._create_fallback_result(query, session_id, str(e)))
        
        return results
    
    def get_intent_confidence_threshold(self) -> float:
        """
        Get the minimum confidence threshold for intent classification.
        
        Returns:
            Confidence threshold value
        """
        return 0.5
    
    def is_high_confidence_intent(self, intent_result: Dict[str, Any]) -> bool:
        """
        Check if the intent classification has high confidence.
        
        Args:
            intent_result: Intent classification result dictionary
            
        Returns:
            True if high confidence, False otherwise
        """
        return intent_result.get("confidence", 0.0) >= self.get_intent_confidence_threshold()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the intent agent configuration.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "agent_type": "Flexible Query Analyzer",
            "llm_provider": self.config.get("provider"),
            "llm_model": self.config.get("model"),
            "analysis_approach": "flexible_semantic_analysis",
            "rigid_categories": False,
            "confidence_threshold": self.get_intent_confidence_threshold(),
            "agent_role": self.agent.role,
            "agent_goal": self.agent.goal,
            "capabilities": [
                "semantic_analysis",
                "query_type_detection",
                "specificity_assessment",
                "keyword_extraction",
                "context_understanding"
            ]
        }