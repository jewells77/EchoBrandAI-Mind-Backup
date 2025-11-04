from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any


class CompetitorScrapeRequest(BaseModel):
    """Request model for scraping competitor data."""

    user_id: str = Field(..., description="user id")
    url: HttpUrl = Field(..., description="Single competitor URL to scrape")


class MatchCondition(BaseModel):
    value: str


class FilterCondition(BaseModel):
    key: str
    match: MatchCondition


class FilterDict(BaseModel):
    must: Optional[List[FilterCondition]] = None
    should: Optional[List[FilterCondition]] = None
    must_not: Optional[List[FilterCondition]] = None

    class Config:
        extra = "forbid"


class DeleteWebsiteEmbeddingsRequest(BaseModel):
    user_id: str = Field(..., description="user id")
    filter_dict: FilterDict = Field(
        ...,
        description="Filter dict for deletion with keys: must, should, must_not. Example: { 'should': [ {'key': 'url', 'match': {'value': 'https://example.com/'} } ] }",
    )
