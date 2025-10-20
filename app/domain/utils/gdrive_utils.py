import re


def extract_query_and_gdrive_links(text: str) -> tuple[str, list[str]]:
    """
    Extract Google Drive links from the text, convert them to direct download links,
    and return the cleaned query + list of image URLs.
    """
    pattern = r"https?://drive\.google\.com/[^\s]+"
    drive_links = re.findall(pattern, text)

    image_urls = []
    for link in drive_links:
        image_urls.append(gdrive_to_download_url(link))

    cleaned_query = re.sub(pattern, "", text).strip()

    return cleaned_query, image_urls


def gdrive_to_download_url(url: str) -> str:
    """
    Convert a Google Drive share/view link to a direct download URL.
    """
    patterns = [
        r"file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"thumbnail\?id=([a-zA-Z0-9_-]+)",
    ]

    file_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break

    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {url}")

    return f"https://drive.google.com/uc?export=download&id={file_id}"
