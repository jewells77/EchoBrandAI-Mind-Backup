from pydantic import BaseModel, Field, HttpUrl


class CompetitorScrapeRequest(BaseModel):
    """Request model for scraping competitor data."""

    user_id: str = Field(..., description="user id")
    url: HttpUrl = Field(..., description="Single competitor URL to scrape")


class DeleteWebsiteEmbeddingsRequest(BaseModel):
    user_id: str = Field(..., description="user id")
    filter_dict: dict = Field(
        ..., description="Filter dict for deletion, e.g. {'url': 'www.abc.com'}"
    )
