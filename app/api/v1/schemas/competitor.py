from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class CompetitorScrapeRequest(BaseModel):
    """Request model for scraping competitor data."""

    competitors: List[str] = Field(..., description="List of competitor URLs to scrape")


class CompetitorInsightsResponse(BaseModel):
    """Response model for competitor insights."""

    competitor_insights: List[Dict[str, Any]] = Field(
        ..., description="List of insights from each competitor"
    )
    content_gaps: List[str] = Field(
        ..., description="Content opportunities the brand could exploit"
    )
    trending_topics: List[str] = Field(
        ..., description="Topics trending across competitor content"
    )
    content_types: List[str] = Field(
        ..., description="Content formats being used by competitors"
    )
    scraping_error: Optional[bool] = Field(
        None, description="Indicates if there were errors during competitor scraping"
    )
    error_details: Optional[List[str]] = Field(
        None, description="Details about scraping errors if any occurred"
    )
