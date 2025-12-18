from pydantic import BaseModel
from typing import List

class RecommendationRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    query: str
    embeddings: List[float]
