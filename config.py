"""
Configuration file for the Redis Voice Agent RAG system.
Contains all necessary settings for Redis, ChromaDB, and API integrations.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys - Set these in your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")

# Redis Configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "decode_responses": True,
    "password": os.getenv("REDIS_PASSWORD", None),
    "socket_connect_timeout": 5,
    "socket_timeout": 5,
}

# Redis Key Patterns
REDIS_KEYS = {
    "chat_session": "chat:{session_id}",
    "intent": "intent:{session_id}",
    "session_meta": "session_meta:{session_id}",
    "summary_cache": "summary_cache:{intent}",
    "docs_cache": "docs_cache:{query_hash}",
    "locks": "locks:{query_hash}",
}

# Redis TTL Settings (in seconds)
REDIS_TTL = {
    "chat_session": 3600,  # 1 hour
    "intent": 1800,        # 30 minutes
    "session_meta": 7200,  # 2 hours
    "summary_cache": 86400, # 24 hours
    "docs_cache": 43200,   # 12 hours
    "locks": 300,          # 5 minutes
}

# ChromaDB Configuration
CHROMADB_CONFIG = {
    "collection_name": "privacy_policy_docs",
    "persist_directory": "./chroma_db",
    "embedding_function": "sentence-transformers",  # Use SentenceTransformers embeddings
    "embedding_model": "all-MiniLM-L6-v2",  # SentenceTransformers model
    "sentence_transformer_model": "all-MiniLM-L6-v2",  # SentenceTransformers model
    "distance_metric": "cosine"
}

# LLM Configuration
LLM_CONFIG = {
    "provider": "openai",  # Options: "openai", "anthropic", "demo"
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1500,
    "timeout": 30
}

# Intent Classification
INTENT_LABELS = [
    "privacy_policy_question",
    "data_usage_inquiry", 
    "contact_information",
    "cookie_policy",
    "data_security",
    "children_privacy",
    "third_party_links",
    "policy_updates",
    "general_inquiry",
    "complaint_or_concern",
    "conversation_ending"  # For thanks, bye, farewell messages
]

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "similarity_threshold": 0.7,
    "max_docs_to_retrieve": 10,
}

# Orchestrator Configuration
ORCHESTRATOR_CONFIG = {
    "max_chat_history": 10,
    "session_timeout": 3600,
    "enable_caching": True,
    "cache_similarity_threshold": 0.8,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "rag_system.log",
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict[str, Any]: Complete configuration settings
    """
    return {
        "api_keys": {
            "openai": OPENAI_API_KEY,
            "anthropic": ANTHROPIC_API_KEY,
        },
        "redis": REDIS_CONFIG,
        "redis_keys": REDIS_KEYS,
        "redis_ttl": REDIS_TTL,
        "chromadb": CHROMADB_CONFIG,
        "llm": LLM_CONFIG,
        "intents": INTENT_LABELS,
        "retrieval": RETRIEVAL_CONFIG,
        "orchestrator": ORCHESTRATOR_CONFIG,
        "logging": LOGGING_CONFIG,
    }

def validate_config() -> bool:
    """
    Validate that all required configuration values are set.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [OPENAI_API_KEY]
    
    for key in required_keys:
        if not key or key.startswith("your-"):
            print(f"Warning: Missing or invalid API key configuration")
            return False
    
    return True