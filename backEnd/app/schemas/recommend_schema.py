"""
Recommendation API Schemas
Pydantic models for request/response validation.
"""
import re
from pydantic import BaseModel, Field
from typing import Optional


# Test type letter to full name mapping
TEST_TYPE_MAPPING = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}


def parse_test_types(test_type_str: Optional[str]) -> list[str]:
    """
    Convert test type letter codes to full names.
    E.g., "P, C" -> ["Personality & Behavior", "Competencies"]
    """
    if not test_type_str:
        return []
    
    # Split by comma and strip whitespace
    letters = [t.strip().upper() for t in test_type_str.split(",")]
    
    # Map to full names
    full_names = []
    for letter in letters:
        if letter in TEST_TYPE_MAPPING:
            full_names.append(TEST_TYPE_MAPPING[letter])
    
    return full_names


def parse_duration(assessment_length: Optional[str]) -> Optional[int]:
    """
    Extract duration in minutes from assessment_length string.
    E.g., "25 minutes" -> 25, "11" -> 11
    """
    if not assessment_length:
        return None
    
    # Try to extract numeric value
    match = re.search(r'(\d+)', str(assessment_length))
    if match:
        return int(match.group(1))
    
    return None


class RecommendationRequest(BaseModel):
    """Request schema for product recommendations."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=50000,  # Support long job descriptions
        description="The search query or job description to find relevant assessments"
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


class AssessmentRecommendation(BaseModel):
    """Schema for a single assessment recommendation."""
    url: str = Field(..., description="Assessment URL")
    name: str = Field(..., description="Assessment name")
    adaptive_support: str = Field(..., description="Adaptive/IRT support (Yes/No)")
    description: Optional[str] = Field(None, description="Assessment description")
    duration_minutes: Optional[int] = Field(None, description="Assessment duration in minutes")
    remote_support: str = Field(..., description="Remote testing support (Yes/No)")
    test_type: list[str] = Field(default_factory=list, description="List of test type categories")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
                "name": "Python (New)",
                "adaptive_support": "No",
                "description": "Multi-choice test that measures the knowledge of Python programming...",
                "duration_minutes": 11,
                "remote_support": "Yes",
                "test_type": ["Knowledge & Skills"]
            }
        }


class RecommendationResponse(BaseModel):
    """Response schema for product recommendations."""
    recommended_assessments: list[AssessmentRecommendation] = Field(
        default_factory=list,
        description="List of recommended assessments sorted by relevance"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "recommended_assessments": [
                    {
                        "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
                        "name": "Python (New)",
                        "adaptive_support": "No",
                        "description": "Multi-choice test that measures the knowledge of Python programming...",
                        "duration_minutes": 11,
                        "remote_support": "Yes",
                        "test_type": ["Knowledge & Skills"]
                    }
                ]
            }
        }
