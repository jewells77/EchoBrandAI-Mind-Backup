from pydantic import BaseModel, Field, HttpUrl


class CompetitorScrapeRequest(BaseModel):
    """Request model for scraping competitor data."""

    url: HttpUrl = Field(..., description="Single competitor URL to scrape")
    user_id: str = Field(..., description="user id")


class DeleteWebsiteEmbeddingsRequest(BaseModel):
    user_id: str = Field(..., description="user id")
    filter_dict: dict = Field(
        ..., description="Filter dict for deletion, e.g. {'url': 'www.abc.com'}"
    )
