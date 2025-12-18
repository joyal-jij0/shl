"""
Recommendation API Endpoint
Provides product recommendations based on natural language queries.
"""
import logging
from fastapi import APIRouter, HTTPException

from app.schemas.recommend_schema import (
    RecommendationRequest,
    RecommendationResponse,
    AssessmentRecommendation,
    parse_test_types,
    parse_duration
)
from app.services.recommend_service import get_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get Assessment Recommendations",
    description="""
    Get AI-powered assessment recommendations based on a natural language query.
    
    The endpoint uses semantic search combined with keyword boosting to find 
    the most relevant assessments based on the query.
    
    **Example queries:**
    - "I need a cognitive ability test for entry-level software developers"
    - "Assessment for leadership skills in senior management"
    - "Quick personality test with remote testing support"
    """,
    responses={
        200: {"description": "Successful response with recommendations"},
        400: {"description": "Invalid request (e.g., empty query)"},
        500: {"description": "Internal server error"}
    }
)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """
    Generate assessment recommendations based on a natural language query.
    """
    try:
        logger.info(f"Received recommendation request: query='{request.query[:50]}...', top_k={request.top_k}")
        
        # Get recommendations from the service
        results = await get_recommendations(
            query=request.query,
            top_k=request.top_k
        )
        
        # Convert SearchResult objects to the expected output format
        recommended_assessments = []
        for result in results:
            product = result.product
            assessment = AssessmentRecommendation(
                url=product.url,
                name=product.name,
                adaptive_support="Yes" if product.adaptive_irt else "No",
                description=product.description,
                duration_minutes=parse_duration(product.assessment_length),
                remote_support="Yes" if product.remote_testing else "No",
                test_type=parse_test_types(product.test_type)
            )
            recommended_assessments.append(assessment)
        
        response = RecommendationResponse(
            recommended_assessments=recommended_assessments
        )
        
        logger.info(f"Returning {len(recommended_assessments)} recommendations")
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
