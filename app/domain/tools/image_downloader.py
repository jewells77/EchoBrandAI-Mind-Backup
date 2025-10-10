import httpx
from app.api.exceptions import APIError


async def download_image_from_url(image_url: str) -> bytes:
    """
    Download an image from a URL and return its bytes.
    Args:
        image_url: The URL of the image to download.
    Returns:
        Image bytes.
    Raises:
        httpx.HTTPStatusError: If the request fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        status_code = getattr(e.response, "status_code", 500)
        raise APIError(
            str(e),
            status_code=status_code,
            public_message=f"Failed to download image from URL: {image_url}.",
        )
    except httpx.RequestError as e:
        raise APIError(
            str(e),
            status_code=500,
            public_message=f"Failed to download image from URL: {image_url}.",
        )

    # Validate content type
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise APIError(
            f"URL does not point to an image. Content-Type: {content_type}",
            status_code=422,
            public_message=f"URL does not point to an image. Content-Type",
        )
    return response.content
