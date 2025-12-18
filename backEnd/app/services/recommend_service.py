"""
Recommendation Service
Orchestrates the recommendation pipeline: embedding generation → hybrid search.
"""
import os
import logging

from app.services.embedding_service import get_query_embedding
from app.services.vector_search_service import hybrid_search, SearchResult

logger = logging.getLogger(__name__)

# Database path - use absolute path based on project structure
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "shl_products.db"
)


async def get_recommendations(
    query: str,
    top_k: int = 10,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> list[SearchResult]:
    """
    Get product recommendations based on a natural language query.
    
    This function orchestrates the full recommendation pipeline:
    1. Generate embedding for the query using Azure OpenAI
    2. Perform hybrid search (semantic + keyword) against stored product embeddings
    3. Return top-k most relevant products
    
    Args:
        query: Natural language query describing the assessment needs
        top_k: Number of top results to return
        semantic_weight: Weight for semantic similarity
        keyword_weight: Weight for keyword matching
    
    Returns:
        List of SearchResult objects sorted by relevance
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
    
    # Step 2: Perform hybrid search
    try:
        logger.debug(f"Performing hybrid search (top_k={top_k})...")
        results = hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            db_path=DB_PATH,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        logger.info(f"Found {len(results)} recommendations for query")
        return results
    
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise RuntimeError(f"Failed to perform hybrid search: {e}")
