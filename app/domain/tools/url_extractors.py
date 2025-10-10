import re
from typing import List, Tuple
from app.config import settings
from app.api.exceptions import APIError

# Regex for matching Azure Blob Storage image links (generic, will filter by allowed base URLs later)
AZURE_BLOB_IMAGE_REGEX = r"https://[\w.-]+\.blob\.core\.windows\.net/[\w/-]+/[^\s'\"<>]+\.(?:png|jpg|jpeg|gif|webp)"


def extract_query_and_azure_media_links(
    text: str, max_links: int = 3
) -> Tuple[str, List[str]]:
    """
    Extracts allowed Azure Blob Storage image URLs from input text, removes them from the query, and enforces a limit.
    Only URLs that start with any configured allowed base URLs are accepted.

    Args:
        text (str): The input user query text.
        max_links (int): Maximum number of image URLs allowed.

    Returns:
        Tuple[str, List[str]]: (query_without_urls, list_of_image_urls)
    Raises:
        APIError: If more than max_links URLs are found, or any URL is not from an allowed base.
    """
    allowed_bases = settings.ALLOWED_AZURE_BLOB_BASE_URLS
    image_urls = re.findall(AZURE_BLOB_IMAGE_REGEX, text, re.IGNORECASE)
    filtered_urls = [
        url for url in image_urls if any(url.startswith(base) for base in allowed_bases)
    ]
    if len(filtered_urls) != len(image_urls):
        raise APIError(
            "One or more image URLs are not from an allowed Azure Blob Storage base URL.",
            status_code=400,
            public_message="We do not support the provided image URL.",
        )
    if len(filtered_urls) > max_links:
        raise APIError(
            f"A maximum of {max_links} image URLs are allowed.", status_code=400
        )
    # Remove found URLs from query (replace with '')
    query = text
    for url in filtered_urls:
        query = query.replace(url, "")
    # Clean extra whitespace
    query = " ".join(query.split())
    return query, filtered_urls
