"""
Database Service
Handles all database operations for product retrieval.
"""
import sqlite3
import json
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default database path (relative to project root)
DEFAULT_DB_PATH = "shl_products.db"


@dataclass
class Product:
    """Represents a product from the database."""
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
    embedding: Optional[list[float]]


def parse_embedding(embedding_str: str) -> Optional[list[float]]:
    """
    Parse embedding string from database into a list of floats.
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


def get_all_products(db_path: str = DEFAULT_DB_PATH) -> list[Product]:
    """
    Retrieve all products with their embeddings from the database.
    
    Args:
        db_path: Path to the SQLite database
    
    Returns:
        List of Product objects with parsed embeddings
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
        
        rows = cursor.fetchall()
        conn.close()
        
        products = []
        for row in rows:
            (
                prod_id, name, url, remote_testing, adaptive_irt,
                test_type, description, job_levels, languages,
                assessment_length, embedding_str
            ) = row
            
            embedding = parse_embedding(embedding_str)
            if embedding is None:
                logger.debug(f"Skipping product {prod_id}: invalid embedding")
                continue
            
            product = Product(
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
                embedding=embedding
            )
            products.append(product)
        
        logger.info(f"Retrieved {len(products)} products with valid embeddings from database")
        return products
        
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching products: {e}")
        raise RuntimeError(f"Database error: {e}")


def get_product_by_id(product_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Product]:
    """
    Retrieve a single product by ID.
    
    Args:
        product_id: The product ID to fetch
        db_path: Path to the SQLite database
    
    Returns:
        Product object or None if not found
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
            WHERE id = ?
        """, (product_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        (
            prod_id, name, url, remote_testing, adaptive_irt,
            test_type, description, job_levels, languages,
            assessment_length, embedding_str
        ) = row
        
        return Product(
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
            embedding=parse_embedding(embedding_str)
        )
        
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching product {product_id}: {e}")
        raise RuntimeError(f"Database error: {e}")
