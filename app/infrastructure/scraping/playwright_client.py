from typing import Dict, Any, Optional, List
import asyncio
import httpx
from urllib.parse import urlparse


from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

from app.core.logger import get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

    # Removed _setup_browser and _cleanup; handled by LangChain utils

    async def _fetch_with_httpx(self, url: str) -> Dict[str, Any]:
        """
        Fallback method to fetch content using httpx when Playwright fails.

        Args:
            url: The URL to scrape

        Returns:
            Dict containing basic content and metadata
        """
        logger.info(f"Using httpx fallback for {url}")
        try:
            headers = {"User-Agent": self.user_agent}
            async with httpx.AsyncClient(
                headers=headers, timeout=self.timeout / 1000
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Get domain from URL for title fallback
                domain = urlparse(url).netloc

                return {
                    "url": url,
                    "title": f"Content from {domain}",
                    "text_content": response.text[
                        :10000
                    ],  # Truncate text to avoid massive content
                    "meta_description": "",
                    "full_html": response.text,
                }
        except Exception as e:
            logger.error(f"Error in httpx fallback for {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e),
                "text_content": f"Failed to fetch content from {url}. Error: {str(e)}",
                "title": f"Error fetching {url}",
                "meta_description": "",
                "full_html": "",
            }

    async def fetch_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch content from a given URL using LangChain's Playwright utilities for full browser control.

        Args:
            url: The URL to scrape

        Returns:
            Dict containing page content, title, and metadata
        """
        if self.use_fallback:
            return await self._fetch_with_httpx(url)

        try:
            browser = create_async_playwright_browser(headless=self.headless)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")

            text_content = await page.evaluate(
                """
                    () => {
                        const removeSelectors = [
                            'script', 'style', 'noscript', 'iframe', 'img', 'svg', 'button', 
                            'nav', 'footer', 'form'
                        ];
                        for (const selector of removeSelectors) {
                            document.querySelectorAll(selector).forEach(el => el.remove());
                        }

                        const main = document.querySelector('main') 
                                || document.querySelector('article') 
                                || document.querySelector('#content') 
                                || document.querySelector('body');

                        if (!main) return '';

                        function getText(el) {
                            let text = '';
                            for (const child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE) {
                                    text += child.textContent.trim() + ' ';
                                } else if (child.nodeType === Node.ELEMENT_NODE) {
                                    const tag = child.tagName.toLowerCase();
                                    const childText = getText(child);

                                    if (['p', 'div', 'section', 'article'].includes(tag)) {
                                        text += childText + '\\n\\n'; // double break for paragraphs
                                    } else if (['h1', 'h2', 'h3'].includes(tag)) {
                                        text += '\\n' + childText.toUpperCase() + '\\n';
                                    } else {
                                        text += childText;
                                    }
                                }
                            }
                            return text;
                        }

                        return getText(main).replace(/\\s+\\n/g, '\\n').trim();
                    }
                """
            )
            await context.close()
            await browser.close()
            return {"url": url, "text_content": text_content}
        except Exception as e:
            logger.error(f"Error scraping {url} with LangChain Playwright: {str(e)}")
            return await self._fetch_with_httpx(url)

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
