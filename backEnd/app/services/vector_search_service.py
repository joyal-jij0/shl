"""
Vector Search Service
Provides hybrid search combining semantic embeddings with keyword boosting.
"""
import math
import re
import logging
from typing import Optional
from dataclasses import dataclass
from collections import Counter

from app.services.database_service import Product, get_all_products, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# Hybrid search weights (must sum to 1.0)
SEMANTIC_WEIGHT = 0.6  # Weight for cosine similarity
KEYWORD_WEIGHT = 0.4   # Weight for keyword matching

# Field-specific boost multipliers for keyword matching
FIELD_BOOSTS = {
    "name": 3.0,           # Product name is most important
    "test_type": 2.5,      # Test type is very relevant
    "job_levels": 2.0,     # Job level matching is important
    "description": 1.0,    # Description has base weight
}

# Common stopwords to ignore in keyword matching
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "test", "tests", "testing", "assessment", "assessments"
}


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    product: Product
    similarity_score: float


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase words, removing punctuation and stopwords.
    """
    if not text:
        return []
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.
    Uses log-normalized TF: 1 + log(count)
    """
    if not tokens:
        return {}
    
    counts = Counter(tokens)
    return {term: 1 + math.log(count) for term, count in counts.items()}


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity = (A · B) / (||A|| × ||B||)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}")
    
    if len(vec_a) == 0:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def compute_keyword_score(query_tokens: list[str], product: Product) -> float:
    """
    Compute keyword matching score between query and product fields.
    Uses a BM25-inspired approach with field boosting.
    """
    if not query_tokens:
        return 0.0
    
    query_tf = compute_tf(query_tokens)
    total_score = 0.0
    max_possible_score = 0.0
    
    # Score each field with its boost
    fields = {
        "name": product.name or "",
        "test_type": product.test_type or "",
        "job_levels": product.job_levels or "",
        "description": product.description or "",
    }
    
    for field_name, field_value in fields.items():
        field_tokens = tokenize(field_value)
        if not field_tokens:
            continue
        
        field_tf = compute_tf(field_tokens)
        boost = FIELD_BOOSTS.get(field_name, 1.0)
        
        for term, query_weight in query_tf.items():
            max_possible_score += query_weight * boost
            
            if term in field_tf:
                total_score += query_weight * field_tf[term] * boost
            else:
                for field_term in field_tf:
                    if term in field_term or field_term in term:
                        total_score += query_weight * field_tf[field_term] * boost * 0.5
                        break
    
    # Boolean feature matching
    if "remote" in query_tokens and product.remote_testing:
        total_score += 2.0
        max_possible_score += 2.0
    
    if ("adaptive" in query_tokens or "irt" in query_tokens) and product.adaptive_irt:
        total_score += 2.0
        max_possible_score += 2.0
    
    if max_possible_score > 0:
        return min(1.0, total_score / max_possible_score)
    
    return 0.0


def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    semantic_weight: float = SEMANTIC_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT
) -> list[SearchResult]:
    """
    Perform hybrid search combining semantic similarity with keyword boosting.
    
    Args:
        query: Original query text for keyword matching
        query_embedding: The embedding vector of the query
        top_k: Number of top results to return
        db_path: Path to the SQLite database
        semantic_weight: Weight for semantic similarity
        keyword_weight: Weight for keyword matching
    
    Returns:
        List of SearchResult objects sorted by combined score
    """
    if not query_embedding:
        logger.warning("Empty query embedding provided")
        return []
    
    query_tokens = tokenize(query)
    logger.debug(f"Query tokens: {query_tokens}")
    
    products = get_all_products(db_path)
    
    if not products:
        logger.warning("No products found in database")
        return []
    
    results: list[tuple[SearchResult, float]] = []
    
    for product in products:
        if not product.embedding:
            continue
        
        # Semantic similarity
        try:
            semantic_score = cosine_similarity(query_embedding, product.embedding)
            semantic_score_normalized = (semantic_score + 1) / 2
        except ValueError as e:
            logger.warning(f"Skipping product {product.id}: {e}")
            continue
        
        # Keyword matching
        keyword_score = compute_keyword_score(query_tokens, product)
        
        # Combined score
        combined_score = (
            semantic_weight * semantic_score_normalized +
            keyword_weight * keyword_score
        )
        
        result = SearchResult(
            product=product,
            similarity_score=round(combined_score, 6)
        )
        
        results.append((result, combined_score))
        
        logger.debug(
            f"Product '{product.name}': semantic={semantic_score:.4f}, "
            f"keyword={keyword_score:.4f}, combined={combined_score:.4f}"
        )
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = [r[0] for r in results[:top_k]]
    
    logger.info(f"Hybrid search completed: found {len(top_results)} results (top {top_k})")
    
    return top_results
