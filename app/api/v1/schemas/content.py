from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, List, Optional


class BrandDetails(BaseModel):
    """Brand details for content generation."""

    name: str = Field(..., description="Brand name")
    description: str = Field(..., description="Brand description")
    values: Optional[List[str]] = Field(default=[], description="Brand values")
    industry: str = Field(..., description="Industry the brand operates in")
    mission_statement: Optional[str] = Field(
        None, description="Brand mission statement"
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        None, description="Additional brand information"
    )


class ContentGenerationRequest(BaseModel):
    """Request model for content generation."""

    brand_details: BrandDetails = Field(..., description="Details about the brand")
    competitors: Optional[List[str]] = Field(
        default=[], description="List of competitor URLs or names"
    )
    content_request: str = Field(..., description="Content request or brief")
    guidelines: Optional[Dict[str, Any]] = Field(
        None, description="Content guidelines including tone, target audience, etc."
    )


class ContentGenerationResponse(BaseModel):
    """Response model for content generation."""

    brand_profile: Dict[str, Any] = Field(
        ..., description="Brand profile with extracted DNA"
    )
    competitor_insights: Dict[str, Any] = Field(
        ..., description="Insights from competitor analysis"
    )
    content_strategy: Dict[str, Any] = Field(
        ..., description="Content strategy recommendations"
    )
    final_content: Dict[str, Any] = Field(..., description="Final generated content")
    job_id: Optional[str] = Field(
        None, description="Job ID for asynchronous processing"
    )
