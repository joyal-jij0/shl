import os
import logging
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

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

async def get_query_embedding(text: str) -> list[float]:
    """
    Generates an embedding for the given text using Azure OpenAI asynchronously.
    """
    if not client:
        logger.error("Azure OpenAI client is not initialized.")
        raise RuntimeError("Azure OpenAI client is not initialized.")

    try:
        deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not deployment_name:
            logger.error("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not found in environment variables.")
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not configured.")

        # Standardize text
        text = text.replace("\n", " ").strip()
        if not text:
            return []
            
        response = await client.embeddings.create(
            input=[text], 
            model=deployment_name
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise e
