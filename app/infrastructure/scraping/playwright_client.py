from typing import Dict, Any, Optional, List
import asyncio
import httpx
from urllib.parse import urlparse
from playwright.async_api import async_playwright

from app.core.logger import get_logger

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

    async def _setup_browser(self):
        """Set up and return Playwright browser instance."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self.headless)
        context = await browser.new_context(
            user_agent=self.user_agent, viewport={"width": 1280, "height": 800}
        )
        return playwright, browser, context

    async def _cleanup(self, playwright, browser):
        """Clean up Playwright resources."""
        await browser.close()
        await playwright.stop()

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
        Fetch content from a given URL.

        Args:
            url: The URL to scrape

        Returns:
            Dict containing page content, title, and metadata
        """
        # Use httpx fallback if configured
        if self.use_fallback:
            return await self._fetch_with_httpx(url)

        # Try to use Playwright
        try:
            playwright, browser, context = await self._setup_browser()
        except NotImplementedError:
            logger.warning(
                "Playwright not supported on this system, using httpx fallback"
            )
            # If Playwright fails with NotImplementedError, fall back to httpx
            return await self._fetch_with_httpx(url)
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {str(e)}")
            # Any other exception during playwright setup, fall back to httpx
            return await self._fetch_with_httpx(url)

        try:
            page = await context.new_page()
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")

            # Extract page title
            title = await page.title()

            # Extract page content
            body_content = await page.content()

            # Extract main text content
            text_content = await page.evaluate(
                """() => {
                // Remove script and style elements
                const elements = document.querySelectorAll('script, style, noscript, iframe, img');
                for (const element of elements) {
                    element.remove();
                }
                
                // Extract main content (prioritize main, article, or body)
                const main = document.querySelector('main') || 
                             document.querySelector('article') || 
                             document.querySelector('body');
                             
                return main ? main.textContent.replace(/\\s+/g, ' ').trim() : '';
            }"""
            )

            # Extract metadata
            meta_description = await page.evaluate(
                """() => {
                const metaDesc = document.querySelector('meta[name="description"]');
                return metaDesc ? metaDesc.getAttribute('content') : '';
            }"""
            )

            return {
                "url": url,
                "title": title,
                "text_content": text_content,
                "meta_description": meta_description,
                "full_html": body_content,
            }

        except Exception as e:
            logger.error(f"Error scraping {url} with Playwright: {str(e)}")
            # If Playwright execution fails, try the fallback
            return await self._fetch_with_httpx(url)

        finally:
            try:
                await self._cleanup(playwright, browser)
            except Exception as e:
                logger.error(f"Error during Playwright cleanup: {str(e)}")

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
