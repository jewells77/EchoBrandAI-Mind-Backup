from typing import Dict, Any, Optional, List
import asyncio
from app.api.exceptions import APIError

from bs4 import BeautifulSoup

# from langchain_community.tools.playwright.utils import (
#     create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
# )

from app.core.logger import get_logger
from playwright.async_api import async_playwright

# Patch event loop for environments like Jupyter
import nest_asyncio

nest_asyncio.apply()
logger = get_logger(__name__)


class PlaywrightScraper:
    """Client for scraping web content using Playwright."""

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        user_agent: Optional[str] = None,
        use_fallback: bool = False,
    ):
        """
        Initialize the Playwright scraper.

        Args:
            headless: Whether to run browser in headless mode
            timeout: Navigation timeout in milliseconds
            user_agent: Custom user agent string
            use_fallback: Whether to use fallback method instead of Playwright
        """
        self.headless = headless
        self.timeout = timeout
        self.use_fallback = use_fallback
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )

    async def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch detailed text content from a given URL using Playwright and BeautifulSoup.

        Args:
            url: The URL to scrape

        Returns:
            Dict containing page title, meta description, and extracted text
        """

        browser = None
        context = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(url, timeout=self.timeout, wait_until="networkidle")

                # Scroll page to trigger lazy-loaded content
                await page.evaluate(
                    """
                        () => {
                            return new Promise(resolve => {
                                let totalHeight = 0;
                                const distance = 400;
                                const timer = setInterval(() => {
                                    const scrollHeight = document.body.scrollHeight;
                                    window.scrollBy(0, distance);
                                    totalHeight += distance;
                                    if(totalHeight >= scrollHeight){
                                        clearInterval(timer);
                                        setTimeout(resolve, 500); // wait for lazy content
                                    }
                                }, 200);
                            });
                        }
                        """
                )

                # Get the full HTML after scrolling
                html = await page.content()
                title = await page.title()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted tags
            for tag in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "iframe",
                    "svg",
                    "button",
                    "nav",
                    "footer",
                    "form",
                ]
            ):
                tag.decompose()

            # Extract text
            main = soup.find(["main", "article", "section", "body"])
            text_content = main.get_text(separator="\n") if main else ""
            cleaned_text = "\n".join(
                [line.strip() for line in text_content.splitlines() if line.strip()]
            )

            # Get meta description if exists
            meta_tag = soup.find("meta", attrs={"name": "description"})
            meta_description = (
                meta_tag["content"] if meta_tag and meta_tag.has_attr("content") else ""
            )

            return {
                "url": url,
                "title": title or "",
                "meta_description": meta_description,
                "text_content": cleaned_text,
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise APIError("Error scraping the URL", status_code=500)

    async def fetch_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch content from multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries containing scraped content
        """
        tasks = [self.fetch_content(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results
