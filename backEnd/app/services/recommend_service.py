"""
Recommendation Service
Orchestrates the recommendation pipeline: embedding generation → vector search.
"""
import os
import logging
from typing import Optional

from app.services.embedding_service import get_query_embedding
from app.services.vector_search_service import vector_search, ProductResult

logger = logging.getLogger(__name__)

# Database path - use absolute path based on project structure
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "shl_products.db"
)


async def get_recommendations(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.0
) -> list[ProductResult]:
    """
    Get product recommendations based on a natural language query.
    
    This function orchestrates the full recommendation pipeline:
    1. Generate embedding for the query using Azure OpenAI
    2. Perform vector similarity search against stored product embeddings
    3. Return top-k most relevant products
    
    Args:
        query: Natural language query describing the assessment needs
        top_k: Number of top results to return (default: 10)
        similarity_threshold: Minimum similarity score to include (default: 0.0)
    
    Returns:
        List of ProductResult objects sorted by relevance (highest first)
    
    Raises:
        RuntimeError: If embedding generation fails
        ValueError: If query is empty or invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    logger.info(f"Processing recommendation request for query: '{query[:100]}...'")
    
    # Step 1: Generate embedding for the query
    try:
        logger.debug("Generating query embedding...")
        query_embedding = await get_query_embedding(query)
        
        if not query_embedding:
            raise RuntimeError("Failed to generate embedding: empty result")
        
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")
    
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    # Step 2: Perform vector similarity search
    try:
        logger.debug(f"Performing vector search (top_k={top_k})...")
        results = vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            db_path=DB_PATH,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"Found {len(results)} recommendations for query")
        return results
    
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise RuntimeError(f"Failed to perform vector search: {e}")
