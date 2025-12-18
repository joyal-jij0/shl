"""
Recommendation API Endpoint
Provides product recommendations based on natural language queries.
"""
import logging
from fastapi import APIRouter, HTTPException

from app.schemas.recommend_schema import (
    RecommendationRequest,
    RecommendationResponse,
    ProductRecommendation
)
from app.services.recommend_service import get_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get Product Recommendations",
    description="""
    Get AI-powered product recommendations based on a natural language query.
    
    The endpoint uses semantic search to find the most relevant assessment products
    based on the query. It generates embeddings for the query and performs cosine
    similarity matching against pre-computed product embeddings.
    
    **Example queries:**
    - "I need a cognitive ability test for entry-level software developers"
    - "Assessment for leadership skills in senior management"
    - "Quick personality test with remote testing support"
    """,
    responses={
        200: {
            "description": "Successful response with recommendations",
            "model": RecommendationResponse
        },
        400: {
            "description": "Invalid request (e.g., empty query)"
        },
        500: {
            "description": "Internal server error during recommendation generation"
        }
    }
)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """
    Generate product recommendations based on a natural language query.
    
    Args:
        request: RecommendationRequest containing the query and optional top_k
    
    Returns:
        RecommendationResponse with ranked product recommendations
    """
    try:
        logger.info(f"Received recommendation request: query='{request.query[:50]}...', top_k={request.top_k}")
        
        # Get recommendations from the service
        results = await get_recommendations(
            query=request.query,
            top_k=request.top_k
        )
        
        # Convert ProductResult objects to Pydantic models
        recommendations = [
            ProductRecommendation(
                id=result.id,
                name=result.name,
                url=result.url,
                remote_testing=result.remote_testing,
                adaptive_irt=result.adaptive_irt,
                test_type=result.test_type,
                description=result.description,
                job_levels=result.job_levels,
                languages=result.languages,
                assessment_length=result.assessment_length,
                similarity_score=result.similarity_score
            )
            for result in results
        ]
        
        response = RecommendationResponse(
            success=True,
            query=request.query,
            total_results=len(recommendations),
            recommendations=recommendations
        )
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        return response
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"Recommendation service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.exception(f"Unexpected error in recommendation endpoint: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
