"""
Embedding Service
Generates embeddings using Azure OpenAI with support for long texts via chunking.
"""
import os
import math
import logging
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Token limits (text-embedding-ada-002 has ~8191 token limit)
# Using conservative estimate: ~4 chars per token
MAX_CHARS = 25000  # ~6250 tokens, leaving buffer
CHUNK_SIZE = 20000  # ~5000 tokens per chunk
CHUNK_OVERLAP = 2000  # ~500 tokens overlap for context

# Initialize Azure OpenAI client
try:
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    client = None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for processing long documents.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence boundary (., !, ?)
        # Look back from the end to find a good break point
        break_point = end
        for i in range(end, max(start + chunk_size // 2, start), -1):
            if text[i] in '.!?\n':
                break_point = i + 1
                break
        
        chunks.append(text[start:break_point])
        start = break_point - overlap  # Overlap for context
    
    logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """
    Average multiple embeddings into a single embedding.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        Averaged embedding vector (normalized)
    """
    if not embeddings:
        return []
    
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Calculate element-wise average
    dimensions = len(embeddings[0])
    averaged = [0.0] * dimensions
    
    for embedding in embeddings:
        for i, val in enumerate(embedding):
            averaged[i] += val
    
    # Divide by count
    count = len(embeddings)
    averaged = [val / count for val in averaged]
    
    # Normalize the averaged embedding (L2 normalization)
    magnitude = math.sqrt(sum(x * x for x in averaged))
    if magnitude > 0:
        averaged = [x / magnitude for x in averaged]
    
    return averaged


async def get_single_embedding(text: str, deployment_name: str) -> list[float]:
    """
    Generate embedding for a single text chunk.
    """
    response = await client.embeddings.create(
        input=[text],
        model=deployment_name
    )
    return response.data[0].embedding


async def get_query_embedding(text: str) -> list[float]:
    """
    Generates an embedding for the given text using Azure OpenAI.
    
    For long texts that exceed the token limit, the text is split into
    overlapping chunks, each chunk is embedded separately, and the
    embeddings are averaged together.
    
    Args:
        text: The text to embed (can be a long JD or query)
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        RuntimeError: If Azure OpenAI client is not initialized
        ValueError: If deployment name is not configured
    """
    if not client:
        logger.error("Azure OpenAI client is not initialized.")
        raise RuntimeError("Azure OpenAI client is not initialized.")

    try:
        deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not deployment_name:
            logger.error("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not found in environment variables.")
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not configured.")

        # Clean and standardize text
        text = text.replace("\n", " ").strip()
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        if not text:
            return []
        
        # Check if text is short enough for single embedding
        if len(text) <= MAX_CHARS:
            logger.debug(f"Generating single embedding for text ({len(text)} chars)")
            return await get_single_embedding(text, deployment_name)
        
        # Text is too long - use chunking
        logger.info(f"Text too long ({len(text)} chars), using chunking strategy")
        chunks = chunk_text(text)
        
        # Generate embeddings for each chunk
        embeddings = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Generating embedding for chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            embedding = await get_single_embedding(chunk, deployment_name)
            embeddings.append(embedding)
        
        # Average the embeddings
        averaged_embedding = average_embeddings(embeddings)
        logger.info(f"Generated averaged embedding from {len(chunks)} chunks")
        
        return averaged_embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise e
