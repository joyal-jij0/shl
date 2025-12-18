"""
Vector Search Service
Provides cosine similarity search functionality for product embeddings.
"""
import sqlite3
import json
import math
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default database path (relative to project root)
DEFAULT_DB_PATH = "shl_products.db"


@dataclass
class ProductResult:
    """Represents a product search result with similarity score."""
    id: int
    name: str
    url: str
    remote_testing: Optional[bool]
    adaptive_irt: Optional[bool]
    test_type: Optional[str]
    description: Optional[str]
    job_levels: Optional[str]
    languages: Optional[str]
    assessment_length: Optional[str]
    similarity_score: float


def normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalize a vector to unit length.
    This is crucial for accurate cosine similarity computation.
    """
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity = (A · B) / (||A|| × ||B||)
    
    For normalized vectors, this simplifies to just the dot product.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}")
    
    if len(vec_a) == 0:
        return 0.0
    
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # Compute magnitudes
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def get_all_products_with_embeddings(db_path: str = DEFAULT_DB_PATH) -> list[tuple]:
    """
    Retrieve all products with their embeddings from the database.
    
    Returns:
        List of tuples containing product data and embeddings
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id, name, url, remote_testing, adaptive_irt, 
                test_type, description, job_levels, languages, 
                assessment_length, embedding
            FROM products
            WHERE embedding IS NOT NULL AND embedding != ''
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        logger.info(f"Retrieved {len(results)} products with embeddings from database")
        return results
        
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching products: {e}")
        raise RuntimeError(f"Database error: {e}")


def parse_embedding(embedding_str: str) -> Optional[list[float]]:
    """
    Parse embedding string from database into a list of floats.
    Handles JSON-encoded embeddings.
    """
    if not embedding_str:
        return None
    
    try:
        embedding = json.loads(embedding_str)
        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
            return embedding
        return None
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse embedding: {e}")
        return None


def vector_search(
    query_embedding: list[float],
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    similarity_threshold: float = 0.0
) -> list[ProductResult]:
    """
    Perform vector similarity search to find the most relevant products.
    
    Args:
        query_embedding: The embedding vector of the query
        top_k: Number of top results to return (default: 10)
        db_path: Path to the SQLite database
        similarity_threshold: Minimum similarity score to include (default: 0.0)
    
    Returns:
        List of ProductResult objects sorted by similarity score (descending)
    """
    if not query_embedding:
        logger.warning("Empty query embedding provided")
        return []
    
    # Fetch all products with embeddings
    products = get_all_products_with_embeddings(db_path)
    
    if not products:
        logger.warning("No products with embeddings found in database")
        return []
    
    results: list[tuple[ProductResult, float]] = []
    
    for product in products:
        (
            prod_id, name, url, remote_testing, adaptive_irt,
            test_type, description, job_levels, languages,
            assessment_length, embedding_str
        ) = product
        
        # Parse the stored embedding
        stored_embedding = parse_embedding(embedding_str)
        
        if stored_embedding is None:
            logger.debug(f"Skipping product {prod_id}: invalid embedding")
            continue
        
        # Calculate cosine similarity
        try:
            similarity = cosine_similarity(query_embedding, stored_embedding)
        except ValueError as e:
            logger.warning(f"Skipping product {prod_id}: {e}")
            continue
        
        # Apply threshold filter
        if similarity < similarity_threshold:
            continue
        
        # Create ProductResult
        result = ProductResult(
            id=prod_id,
            name=name,
            url=url,
            remote_testing=bool(remote_testing) if remote_testing is not None else None,
            adaptive_irt=bool(adaptive_irt) if adaptive_irt is not None else None,
            test_type=test_type,
            description=description,
            job_levels=job_levels,
            languages=languages,
            assessment_length=assessment_length,
            similarity_score=round(similarity, 6)  # Round for cleaner output
        )
        
        results.append((result, similarity))
    
    # Sort by similarity score (descending) and take top_k
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = [r[0] for r in results[:top_k]]
    
    logger.info(f"Vector search completed: found {len(top_results)} results (top {top_k})")
    
    return top_results
