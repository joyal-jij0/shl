import sqlite3
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DB_PATH = "shl_products.db"

def add_embedding_column():
    """Adds an 'embedding' column to the products table if it doesn't already exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Check if column exists
            cursor.execute("PRAGMA table_info(products)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'embedding' not in columns:
                logger.info("Adding 'embedding' column to 'products' table...")
                cursor.execute("ALTER TABLE products ADD COLUMN embedding TEXT")
                conn.commit()
            else:
                logger.info("'embedding' column already exists.")
    except Exception as e:
        logger.error(f"Error adding column: {e}")

def get_embedding(text, deployment_name=None):
    """Generates an embedding for the given text using Azure OpenAI."""
    try:
        # Get deployment name from env if not provided
        deployment_name = deployment_name or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        
        if not deployment_name:
            logger.error("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not found.")
            return None

        # Standardize text
        text = text.replace("\n", " ").strip()
        if not text:
            return None
            
        response = client.embeddings.create(
            input=[text], 
            model=deployment_name
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def process_single_row(row_id, name, description, job_levels):
    """Worker function to process a single row's embedding."""
    logger.info(f"Generating embedding for: {name}")
    
    # Combine relevant fields for embedding context
    parts = []
    if name: parts.append(f"Product Name: {name}")
    if description: parts.append(f"Description: {description}")
    if job_levels: parts.append(f"Target Job Levels: {job_levels}")
    
    content = "\n".join(parts)
    
    if not content:
        return row_id, None, name
    
    embedding = get_embedding(content)
    return row_id, embedding, name

def process_embeddings(max_workers=10):
    """Fetches products without embeddings and populates them in parallel."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Fetch products that need embeddings
            cursor.execute("""
                SELECT id, name, description, job_levels 
                FROM products 
                WHERE embedding IS NULL
            """)
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("No products without embeddings found.")
                return

            logger.info(f"Processing embeddings for {len(rows)} products using {max_workers} workers...")
            
            # Use ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_row = {
                    executor.submit(process_single_row, rid, name, desc, job): (rid, name) 
                    for rid, name, desc, job in rows
                }
                
                # Process results as they complete
                for future in as_completed(future_to_row):
                    row_id, embedding, name = future.result()
                    
                    if embedding:
                        # Store result (updates are sequential to avoid DB locking issues)
                        try:
                            embedding_json = json.dumps(embedding)
                            cursor.execute(
                                "UPDATE products SET embedding = ? WHERE id = ?", 
                                (embedding_json, row_id)
                            )
                            # We commit periodically or at the end. 
                            # For safety with parallel submissions finishing, we commit here.
                            conn.commit()
                            logger.info(f"Successfully updated embedding for: {name}")
                        except Exception as e:
                            logger.error(f"Error updating DB for {name}: {e}")
                    else:
                        logger.warning(f"Failed to get embedding for: {name}")

    except Exception as e:
        logger.error(f"Error in process_embeddings: {e}")

if __name__ == "__main__":
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.info("Please set them in your .env file.")
    else:
        add_embedding_column()
        process_embeddings()
