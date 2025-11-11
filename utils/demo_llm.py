"""
Demo LLM for testing the RAG system without API keys.
"""

from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional, Dict
import json
import logging

logger = logging.getLogger(__name__)

class DemoLLM(LLM):
    """
    Demo LLM that provides hardcoded responses for testing without API keys.
    """
    
    temperature: float = 0.7
    max_tokens: int = 1000
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response based on the input prompt."""
        
        # Check if this is an intent classification request
        if "intent" in prompt.lower() and "json" in prompt.lower():
            return self._generate_intent_response(prompt)
        
        # Check if this is a RAG response request
        elif "context documents" in prompt.lower() or "answer" in prompt.lower():
            return self._generate_rag_response(prompt)
        
        # Default response
        return "I'm a demo LLM. Please set up your OpenAI API key for full functionality."
    
    def _generate_intent_response(self, message: str) -> str:
        """Generate an intent classification response."""
        
        # Analyze the query to determine intent
        query_lower = message.lower()
        
        if any(word in query_lower for word in ["contact", "phone", "email", "address"]):
            intent = "contact_information"
        elif any(word in query_lower for word in ["cookie", "tracking", "analytics"]):
            intent = "cookie_policy"
        elif any(word in query_lower for word in ["collect", "data", "information", "personal"]):
            intent = "data_usage_inquiry"
        elif any(word in query_lower for word in ["security", "protect", "safe", "encryption"]):
            intent = "data_security"
        elif any(word in query_lower for word in ["policy", "privacy"]):
            intent = "privacy_policy_question"
        elif any(word in query_lower for word in ["third party", "share", "partner"]):
            intent = "third_party_links"
        elif any(word in query_lower for word in ["child", "children", "under 13"]):
            intent = "children_privacy"
        elif any(word in query_lower for word in ["update", "change", "notify"]):
            intent = "policy_updates"
        elif any(word in query_lower for word in ["complaint", "concern", "delete", "remove"]):
            intent = "complaint_or_concern"
        else:
            intent = "general_inquiry"
        
        response = {
            "intent": intent,
            "confidence": 0.85,
            "reasoning": f"Detected keywords suggest this is a {intent.replace('_', ' ')} question",
            "alternative_intents": ["general_inquiry"],
            "timestamp": "2024-11-04T11:00:00Z"
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_rag_response(self, message: str) -> str:
        """Generate a RAG-style response based on context."""
        
        query_lower = message.lower()
        
        # Generate contextual responses based on common queries
        if "contact" in query_lower:
            return """Based on the privacy policy, you can contact TechGropse at:

**Address:** G-02 Block, D 60, D Block, Sector 63, Noida, Uttar Pradesh 201301, India
**Telephone:** +91-9911138726
**Email:** sales@techgropse.com

For any questions about this Privacy Policy or data protection concerns, please reach out using the contact information above."""
        
        elif "cookie" in query_lower:
            return """According to the privacy policy, TechGropse uses cookies for several purposes:

1. **Necessary/Essential Cookies** - Session cookies to enable website functionality and prevent fraud
2. **Policy Notice Cookies** - To remember if you've accepted cookie usage
3. **Functionality Cookies** - To remember your preferences like login details and language settings

You can disable cookies through your browser settings. The website uses cookies to store visitor preferences, track pages accessed, and customize content based on your browser type."""
        
        elif any(word in query_lower for word in ["collect", "data", "information"]):
            return """TechGropse collects the following types of personal information:

**When you contact them directly:**
- Your name
- Email address  
- Phone number
- Message content and attachments

**When you register or inquire:**
- Name
- Company name
- Address
- Email address
- Telephone number

The information collection is clearly stated at the time of collection, and they reserve the right to disclose information to legal or regulatory bodies as required for compliance."""
        
        elif "security" in query_lower:
            return """TechGropse employs industry-standard security measures to protect your data:

- **SSL Encryption**: The site uses Secure Socket Layer (SSL) encryption when accessed via modern browsers (Internet Explorer 11 or higher)
- **Server Authentication**: SSL provides server authentication and data protection
- **Industry Standards**: They employ industry-standard security measures to protect data from loss, misuse, and unauthorized access

These measures help ensure your personal information is protected during transmission and storage."""
        
        elif any(word in query_lower for word in ["share", "third party", "sell"]):
            return """According to the privacy policy, TechGropse has a clear stance on data sharing:

**They do NOT:**
- Sell personal data to third parties
- Share personal data with third parties
- Distribute personal data to third parties

**They MAY engage trusted partners for:**
- Marketing services
- Email or postal communications  
- Data analysis
- Customer support

These partners have limited access to your information and are prohibited from using it for any other purpose. They may also disclose information to legal or regulatory bodies as required for compliance."""
        
        else:
            return """Based on the privacy policy documents, I can help you understand TechGropse's privacy practices. The policy covers data collection, cookie usage, security measures, contact information, and your rights regarding personal data. 

If you have a specific question about their privacy practices, please feel free to ask about:
- What information they collect
- How they use cookies
- Their security measures  
- Contact information
- Data sharing policies
- Children's privacy protection"""
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke method for compatibility with newer LangChain versions."""
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, list) and input:
            # Handle list of messages
            if hasattr(input[-1], 'content'):
                prompt = input[-1].content
            else:
                prompt = str(input[-1])
        else:
            prompt = str(input)
        
        result = self._call(prompt, **kwargs)
        return type('Response', (), {'content': result})()
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "demo_llm"