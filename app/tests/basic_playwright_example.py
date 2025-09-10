"""
Basic Playwright example matching the provided code sample.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playwright.sync_api import sync_playwright


def main():
    """
    Simple example of using Playwright's sync API.
    """
    # initialize playwright
    pw = sync_playwright().start()

    # create Firefox browser object
    browser = pw.firefox.launch(
        # uncomment the lines below if you're using a web driver
        # headless=False,
        # slow_mo=2000
    )

    # create new browser tab
    page = browser.new_page()

    # navigate to web page
    page.goto("https://google.com")

    # web page details and source code
    print(page.content())
    print(page.title())

    # Take a screenshot
    page.screenshot(path="google_screenshot.png")

    # Close browser when done
    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
