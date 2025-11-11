# AI Agentic RAG System

A modular AI Agentic RAG (Retrieval-Augmented Generation) system built with CrewAI, LangChain, Redis, and ChromaDB.

## Architecture

```
USER → ORCHESTRATOR → [Intent Agent + Retrieval Agent] → [Redis + ChromaDB] → OUTPUT
```

### System Components

1. **Orchestrator** (`main.py`): Central controller that coordinates all agents and manages the workflow
2. **Intent Agent** (`agents/intent_agent.py`): CrewAI agent using LangChain LLM to classify user intents
3. **Retrieval Agent** (`agents/retrieval_agent.py`): CrewAI agent for RAG using ChromaDB and Redis caching
4. **Redis Manager** (`utils/redis_manager.py`): Handles all Redis operations for caching and session management
5. **ChromaDB Client** (`vectorstore/chromadb_client.py`): Manages vector storage and retrieval

## Features

- **Low-latency responses** through intelligent Redis caching
- **Session management** with conversation history and context
- **Intent classification** for better query understanding
- **Semantic search** using ChromaDB vector database
- **Distributed locking** to prevent duplicate operations
- **Modular architecture** for easy extension and maintenance
- **Comprehensive logging** and monitoring

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Redis server running locally or accessible remotely
- OpenAI API key (or Anthropic API key)

### 2. Installation

```bash
# Clone or setup the project directory
cd redis_voice_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here  # Optional
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty if no password
```

Or set these as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 4. Install and Start Redis Server

#### Option A: Install Redis with Homebrew (macOS)

```bash
# Install Redis
brew install redis

# Start Redis as a background service (recommended)
brew services start redis

# Verify Redis is running
redis-cli ping  # Should return "PONG"
```

#### Option B: Other Installation Methods

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# On CentOS/RHEL
sudo yum install redis
sudo systemctl start redis
sudo systemctl enable redis

# Or run Redis directly (any system)
redis-server  # Runs in foreground
```

#### Verify Redis Installation

```bash
# Test connection
redis-cli ping  # Should return "PONG"

# Check Redis version
redis-server --version
```

### 5. Initialize ChromaDB with Data

**Important:** Before running the initialization, make sure you have set your OpenAI API key:

```bash
# Set your real OpenAI API key
export OPENAI_API_KEY="your-actual-openai-api-key-here"
```

Run the data initialization script to load the privacy policy into ChromaDB:

```bash
python initialize_data.py --reset
```

This will:
- Use OpenAI's `text-embedding-3-large` model for high-quality embeddings
- Chunk the privacy policy document into manageable pieces
- Store embeddings in ChromaDB for semantic search
- Verify the setup with test queries

### 6. Run the System

```bash
python main.py
```

## Usage Examples

### Basic Query Processing

```python
from main import RAGOrchestrator

# Initialize the orchestrator
orchestrator = RAGOrchestrator()

# Process a query
response = orchestrator.process_query(
    user_query="What information do you collect about users?",
    user_id="user123"  # Optional
)

print(response["response"])
print(f"Intent: {response['intent']['label']}")
print(f"Confidence: {response['intent']['confidence']}")
```

### Session Management

```python
# Create a session and ask multiple questions
session_id = None
queries = [
    "What is your privacy policy?",
    "How do you use cookies?", 
    "Can I delete my data?"
]

for query in queries:
    response = orchestrator.process_query(
        user_query=query,
        session_id=session_id  # Will create new session on first call
    )
    session_id = response["session_id"]  # Reuse session
    print(f"Q: {query}")
    print(f"A: {response['response']}\n")
```

### System Status

```python
# Get system health and statistics
status = orchestrator.get_system_status()
print(f"Redis connected: {status['redis']['connected']}")
print(f"Documents in ChromaDB: {status['chromadb']['collection_info']['count']}")
```

## Configuration Options

### Embedding Model

The system uses OpenAI's **`text-embedding-3-large`** model by default, which provides:
- 3,072 dimensional embeddings
- Superior performance for semantic similarity
- Better understanding of context and nuance
- Optimal for RAG applications

You can modify the embedding model in `config.py` if needed:

```python
CHROMADB_CONFIG = {
    "embedding_model": "text-embedding-3-large",  # or "text-embedding-3-small"
    # ... other config
}
```

### Redis Keys Structure

- `chat:{session_id}` - Conversation history
- `intent:{session_id}` - Last detected intent
- `session_meta:{session_id}` - Session metadata
- `summary_cache:{intent}` - Cached summaries by intent
- `docs_cache:{query_hash}` - Cached document retrievalsl
- `locks:{query_hash}` - Distributed locks

### Intent Categories

- `privacy_policy_question` - General privacy policy questions
- `data_usage_inquiry` - Data collection and usage questions  
- `contact_information` - Company contact requests
- `cookie_policy` - Cookie-related questions
- `data_security` - Security and protection questions
- `children_privacy` - Children's privacy questions
- `third_party_links` - External links questions
- `policy_updates` - Policy change questions
- `general_inquiry` - General business questions
- `complaint_or_concern` - Complaints or concerns

## Performance Optimizations

### Redis Caching Strategy

1. **Document Cache**: Expensive ChromaDB retrievals are cached by query hash
2. **Summary Cache**: Generated summaries are cached by intent type
3. **Session Cache**: Conversation history and metadata cached per session
4. **Distributed Locks**: Prevent duplicate expensive operations

### Latency Reduction

- **Cache-first approach**: Check Redis before expensive operations
- **Intelligent TTL**: Different cache expiration times for different data types
- **Background processing**: Use locks to handle concurrent requests efficiently
- **Optimized embeddings**: Choose appropriate embedding models for speed/accuracy tradeoff

## Monitoring and Logging

The system provides comprehensive logging and monitoring:

```python
# Get cache statistics
cache_stats = orchestrator.redis_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

# Get session information
session_info = orchestrator.get_session_info(session_id)
print(f"Query count: {session_info['metadata']['query_count']}")
```

## Extending the System

### Adding New Intent Types

1. Update `INTENT_LABELS` in `config.py`
2. Add handling logic in the retrieval agent
3. Update prompt templates as needed

### Adding New Data Sources

1. Extend the ChromaDB client to handle new document types
2. Update the data initialization script
3. Add new metadata fields as needed

### Custom Caching Strategies

1. Extend the Redis manager with new cache types
2. Implement custom TTL logic
3. Add cache warming strategies

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**: Ensure Redis server is running and accessible
2. **OpenAI API Errors**: Check API key and rate limits
3. **ChromaDB Errors**: Ensure proper permissions for data directory
4. **Memory Issues**: Adjust chunk sizes and cache TTL values

### Debug Mode

Set logging level to DEBUG in `config.py`:

```python
LOGGING_CONFIG = {
    "level": "DEBUG",
    # ...
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions, please check the logs first, then review the configuration settings.