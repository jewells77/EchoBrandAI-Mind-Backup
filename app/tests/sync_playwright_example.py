"""
Example usage of the synchronous Playwright scraper.
This demonstrates how to use the SyncPlaywrightScraper class.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.infrastructure.scraping.playwright_client import SyncPlaywrightScraper


def main():
    """
    Main function to demonstrate SyncPlaywrightScraper usage.
    """
    # Create scraper instance
    # Use headless=False to see the browser window (useful for debugging)
    scraper = SyncPlaywrightScraper(
        headless=True,  # Set to False to see the browser window
        timeout=30000,  # 30 seconds timeout
        browser_type="firefox",  # Can also use "chrome" or "webkit"
    )

    # Example 1: Fetch content from a URL
    url = "https://google.com"
    result = scraper.fetch_content(url)

    print(f"Title: {result['title']}")
    print(f"Description: {result['meta_description']}")
    print(f"Content length: {len(result['text_content'])} characters")

    # Example 2: Take a screenshot
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshots_dir, "google.png")
    scraper.take_screenshot(url, screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

    # Example 3: Fetch multiple URLs
    urls = ["https://google.com", "https://github.com", "https://stackoverflow.com"]
    results = scraper.fetch_multiple(urls)

    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['url']}")
        print(f"Title: {result['title']}")


if __name__ == "__main__":
    main()
