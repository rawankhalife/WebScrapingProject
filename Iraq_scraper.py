import time
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

BASE_URL = "https://homele.com/properties/for-sale?page={}"


def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"user-agent={UserAgent().random}")
    driver = webdriver.Chrome(options=options)
    return driver


def scrape_listing_pages(pages):
    driver = setup_driver()
    properties = []

    for page in range(1, pages + 1):
        url = BASE_URL.format(page)
        print(f"[LIST] Scraping page {page}: {url}")

        driver.get(url)
        time.sleep(3)

        cards = driver.find_elements(
            By.XPATH,
            '//div[contains(@class, "flex") and contains(@class, "rounded-xl") and contains(@class,"overflow-hidden")]',
        )

        for card in cards:
            try:
                title = card.find_element(By.XPATH, ".//h5").text
            except:
                title = "N/A"
            try:
                bedrooms = card.find_element(
                    By.XPATH, './/div[contains(@class,"item-bedroom")]/span[1]'
                ).text
            except:
                bedrooms = "N/A"
            try:
                bathrooms = card.find_element(
                    By.XPATH, './/div[contains(@class,"item-bathroom")]/span[1]'
                ).text
            except:
                bathrooms = "N/A"
            try:
                number = card.find_element(
                    By.XPATH, './/div[contains(@class,"item-area")]/span[1]'
                ).text
                sqft = round(float(number) * 10.7639, 2)
                area = f"{sqft} sqft"
            except:
                area = "N/A"
            try:
                location = card.find_element(
                    By.XPATH, './/div[contains(@class,"address")]'
                ).text
            except:
                location = "N/A"
            try:
                url = card.find_element(
                    By.XPATH, './/a[contains(@href, "/properties/")]'
                ).get_attribute("href")
            except:
                url = "N/A"

            properties.append(
                {
                    "title": title,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "area": area,
                    "location": location,
                    "url": url,
                }
            )

    driver.quit()
    return properties


def scrape_detail_pages(links):
    driver = setup_driver()
    all_details = []

    for idx, link in enumerate(links, start=1):
        print(f"[DETAIL {idx}] Scraping: {link}")
        driver.get(link)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        try:
            type_elem = soup.select("span.text-gray-600.font-semibold")[1]
            property_type = type_elem.get_text(strip=True)
        except Exception:
            property_type = "N/A"

        try:
            amenities = [
                a.get_text(strip=True)
                for a in soup.select("span.font-semibold.leading-3")
            ]
        except Exception:
            amenities = []
        try:
            overview_elem = soup.select_one(
                "div.block.whitespace-pre-wrap.text-sm.text-gray-500"
            )
            overview_text = overview_elem.get_text(strip=True)
            overview = " ".join(overview_text.split())
        except Exception:
            overview = "N/A"

        try:
            price = soup.select_one(
                'div:has(> span:nth-of-type(1):contains("Price")) span.text-gray-600.font-semibold'
            ).get_text(strip=True)
        except:
            price = "NA"

        all_details.append(
            {
                "url": link,
                "description": overview,
                "amenities": amenities,
                "type": property_type,
                "price": price,
                "country": "Iraq",
            }
        )

    driver.quit()
    return all_details


list_rows = scrape_listing_pages(pages=1)
property_links = [item["url"] for item in list_rows if item["url"] != "N/A"]
detail_rows = scrape_detail_pages(property_links)
df_list = pd.DataFrame(list_rows)
df_detail = pd.DataFrame(detail_rows)
final = df_list.merge(df_detail, on="url", how="left")
final.to_csv("Iraq_Listings_New.csv", index=False)
