"""
Recommendation API Schemas
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class RecommendationRequest(BaseModel):
    """Request schema for product recommendations."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search query to find relevant assessment products"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top results to return (default: 10, max: 50)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I need a cognitive ability test for entry-level software developers",
                "top_k": 10
            }
        }


class ProductRecommendation(BaseModel):
    """Schema for a single product recommendation."""
    id: int = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    url: str = Field(..., description="Product URL")
    remote_testing: Optional[bool] = Field(None, description="Supports remote testing")
    adaptive_irt: Optional[bool] = Field(None, description="Uses Adaptive/IRT technology")
    test_type: Optional[str] = Field(None, description="Type of test")
    description: Optional[str] = Field(None, description="Product description")
    job_levels: Optional[str] = Field(None, description="Suitable job levels")
    languages: Optional[str] = Field(None, description="Available languages")
    assessment_length: Optional[str] = Field(None, description="Assessment duration")
    similarity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Cosine similarity score (0-1, higher is more relevant)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "Verify - Numerical Reasoning",
                "url": "https://www.shl.com/products/verify-numerical-reasoning",
                "remote_testing": True,
                "adaptive_irt": True,
                "test_type": "Ability & Aptitude",
                "description": "Measures the ability to understand numerical data...",
                "job_levels": "Entry Level, Mid-Level",
                "languages": "English, Spanish, French",
                "assessment_length": "25 minutes",
                "similarity_score": 0.8542
            }
        }


class RecommendationResponse(BaseModel):
    """Response schema for product recommendations."""
    success: bool = Field(..., description="Whether the request was successful")
    query: str = Field(..., description="The original search query")
    total_results: int = Field(..., description="Number of results returned")
    recommendations: list[ProductRecommendation] = Field(
        default_factory=list,
        description="List of recommended products sorted by relevance"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "cognitive ability test for developers",
                "total_results": 10,
                "recommendations": [
                    {
                        "id": 1,
                        "name": "Verify - Numerical Reasoning",
                        "url": "https://www.shl.com/products/verify-numerical-reasoning",
                        "remote_testing": True,
                        "adaptive_irt": True,
                        "test_type": "Ability & Aptitude",
                        "description": "Measures the ability to understand numerical data...",
                        "job_levels": "Entry Level, Mid-Level",
                        "languages": "English",
                        "assessment_length": "25 minutes",
                        "similarity_score": 0.8542
                    }
                ]
            }
        }
