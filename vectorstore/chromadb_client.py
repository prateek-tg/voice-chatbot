"""
ChromaDB client for the AI Agentic RAG system.
Handles vector storage and retrieval using LangChain integration.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHROMADB_CONFIG, RETRIEVAL_CONFIG, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class ChromaDBClient:
    """
    ChromaDB client for vector storage and retrieval.
    
    This class provides:
    - Document embedding and storage in ChromaDB
    - Similarity search and retrieval
    - Integration with LangChain's Chroma vectorstore
    - Text chunking and preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            config: Optional ChromaDB configuration dictionary
        """
        self.config = config or CHROMADB_CONFIG
        self.retrieval_config = RETRIEVAL_CONFIG
        
        # Initialize embeddings based on configuration
        self.embeddings = self._initialize_embeddings()
        
        # Initialize ChromaDB client
        self.chroma_client = None
        self.vectorstore = None
        self.collection_name = self.config["collection_name"]
        
        self._initialize_client()
    
    def _initialize_embeddings(self):
        """
        Initialize embedding function based on configuration.
        
        Returns:
            Embedding function instance
        """
        embedding_type = self.config.get("embedding_function", "sentence-transformers")
        embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        
        if embedding_type == "sentence-transformers":
            logger.info(f"Using SentenceTransformers model: {embedding_model}")
            return HuggingFaceEmbeddings(model_name=embedding_model)
        
        elif embedding_type == "openai":
            if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your-"):
                logger.warning("OpenAI API key not configured, falling back to sentence-transformers")
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Use the specified OpenAI embedding model
            return OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model=embedding_model
            )
        
        else:
            logger.warning(f"Unknown embedding type: {embedding_type}, using sentence-transformers")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def _initialize_client(self) -> None:
        """
        Initialize ChromaDB client and vectorstore.
        """
        try:
            # Create persist directory if it doesn't exist
            persist_dir = Path(self.config["persist_directory"])
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize LangChain Chroma vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            logger.info(f"ChromaDB client initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the ChromaDB collection.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            bool: True if documents added successfully, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            # Add documents to vectorstore
            self.vectorstore.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add raw texts to the ChromaDB collection.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            bool: True if texts added successfully, False otherwise
        """
        try:
            if not texts:
                logger.warning("No texts provided to add")
                return False
            
            # Add texts to vectorstore
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info(f"Added {len(texts)} texts to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add texts to ChromaDB: {e}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the ChromaDB collection.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default from config)
            filter: Optional metadata filters
            
        Returns:
            List of Document objects with similar content
        """
        try:
            k = k or self.retrieval_config["top_k"]
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.debug(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query string to search for
            k: Number of results to return (default from config)
            filter: Optional metadata filters
            
        Returns:
            List of tuples (Document, relevance_score)
        """
        try:
            k = k or self.retrieval_config["top_k"]
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Filter by similarity threshold if configured
            threshold = self.retrieval_config.get("similarity_threshold", 0.0)
            if threshold > 0:
                results = [(doc, score) for doc, score in results if score >= threshold]
            
            logger.debug(f"Found {len(results)} similar documents with scores for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search with scores: {e}")
            return []
    
    def create_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """
        Create a LangChain retriever for use in chains.
        
        Args:
            search_type: Type of search ("similarity", "mmr", etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain retriever object
        """
        search_kwargs = search_kwargs or {"k": self.retrieval_config["top_k"]}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (default from config)
            chunk_overlap: Overlap between chunks (default from config)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.retrieval_config["chunk_size"]
        chunk_overlap = chunk_overlap or self.retrieval_config["chunk_overlap"]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        logger.debug(f"Split text into {len(chunks)} chunks")
        
        return chunks
    
    def load_and_chunk_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a document from file and chunk it into Document objects.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of Document objects
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata.update({
                    "source": file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Loaded and chunked document from {file_path} into {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load and chunk document: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            
            info = {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata or {},
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        Delete the ChromaDB collection.
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset (delete and recreate) the ChromaDB collection.
        
        Returns:
            bool: True if reset successfully, False otherwise
        """
        try:
            # Delete existing collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass  # Collection might not exist
            
            # Reinitialize vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config["persist_directory"]
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def close(self) -> None:
        """
        Close the ChromaDB client connections.
        """
        try:
            if self.vectorstore:
                # Persist any changes
                self.vectorstore.persist()
            
            logger.info("ChromaDB client closed")
            
        except Exception as e:
            logger.error(f"Error closing ChromaDB client: {e}")