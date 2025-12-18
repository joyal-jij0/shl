import logging
from app.services.embedding_service import get_query_embedding

logger = logging.getLogger(__name__)

async def get_recommendations(query: str) -> list[float]:
    """
    Process the query to get its vector representation.
    """
    try:
        logger.info(f"Generating embedding for query: {query[:50]}...")
        embedding = await get_query_embedding(query)
        return embedding
    except Exception as e:
        logger.error(f"Error in recommendation service: {e}")
        raise e
