"""Open the Streamlit app in a real browser; click the wake-up button if it is asleep."""
import os
import sys

from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)

    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count() > 0:
        print("App was asleep - clicking wake button")
        wake.click()
        page.wait_for_timeout(90_000)   # give the app time to boot
    else:
        print("App already awake")
        page.wait_for_timeout(20_000)   # stay on the page so the visit counts as traffic

    browser.close()
print("Done")
sys.exit(0)