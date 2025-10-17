from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_scraper
from app.api.v1.schemas.competitor import (
    CompetitorScrapeRequest,
    DeleteWebsiteEmbeddingsRequest,
)
from app.services.competitor_service import CompetitorService
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.api.exceptions import APIError
from app.config import settings
from app.services.competitor_service import delete_qdrant_embeddings_service


router = APIRouter()


@router.post(
    "/website-embeddings",
    summary="Scrape and analyze a single competitor website.",
    description="""
Provide a competitor URL and your user ID to trigger analysis/scraping of that single website. Use this to gather insights for intelligence or pre-scraped content generation data.

Example Parameters:
```json
{
  "user_id": "john123",
  "url": "https://example.com"
}
```
""",
)
async def get_competitor_insights(
    request: CompetitorScrapeRequest,
    scraper: PlaywrightScraper = Depends(get_scraper),
):
    """
    Scrape and analyze a single competitor website to extract insights.

    This endpoint can be used to:
    1. Gather competitive intelligence separate from content generation
    2. Pre-scrape competitor data to use in future content generation requests

    Returns a simple success or raises an HTTPException on error.
    """
    try:
        # Initialize the competitor service
        competitor_service = CompetitorService(scraper)

        # Get competitor insights
        await competitor_service.process_competitor_website(
            url=str(request.url), user_id=request.user_id
        )
        return {
            "message": "Competitor website processed successfully.",
        }

    except APIError as e:
        raise e

    except Exception as e:
        raise APIError(
            str(e),
            status_code=500,
            public_message="Something went wrong.",
        )


@router.delete(
    "/website-embeddings",
    summary="Delete website embeddings from Qdrant",
    description="""
The `filter_dict` supports the following keys: `must`, `should`, and `must_not` for matching different query conditions.

When specifying URLs for deletion, please add every type of value variant so none are missed—for example, both `https://example.com/` and `https://example.com`.

Example Parameters:
```json
{
    "user_id": "john123",
    "filter_dict": {
        "should": [
            {"key": "url", "match": {"value": "https://example.com/"}},
            {"key": "url", "match": {"value": "https://example.com"}}
        ]
    }
}
```
""",
)
async def delete_qdrant_embeddings(request: DeleteWebsiteEmbeddingsRequest):
    try:
        collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION
        must_conditions = [{"key": "user_id", "match": {"value": request.user_id}}]
        if "must" in request.filter_dict:
            must_conditions.extend(request.filter_dict["must"])
        filter_with_user = {"must": must_conditions}
        if "should" in request.filter_dict:
            filter_with_user["should"] = request.filter_dict["should"]
        if "must_not" in request.filter_dict:
            filter_with_user["must_not"] = request.filter_dict["must_not"]
        await delete_qdrant_embeddings_service(collection_name, filter_with_user)
        return {"message": "Embeddings deleted successfully"}
    except APIError as e:
        raise e
    except Exception as e:
        raise APIError(
            str(e),
            status_code=500,
            public_message="Something went wrong.",
        )
