from fastapi import APIRouter, HTTPException
from app.schemas import RecommendationRequest, RecommendationResponse
from app.services.recommend_service import get_recommendations

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Endpoint to get embeddings for a search query.
    """
    try:
        embeddings = await get_recommendations(request.query)
        return RecommendationResponse(
            query=request.query,
            embeddings=embeddings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
