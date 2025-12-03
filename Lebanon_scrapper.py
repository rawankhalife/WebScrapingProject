import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

BASE_URL = "https://www.realestate.com.lb/en/buy-properties-lebanon?pg={}"


def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    ua = UserAgent()
    options.add_argument(f"user-agent={ua.random}")
    driver = webdriver.Chrome(options=options)
    return driver


def scrape_features(pages):
    driver = setup_driver()
    wait = WebDriverWait(driver, 60)
    properties = []
    for page in range(1, pages + 1):
        url = BASE_URL.format(page)
        print(f"Scraping page {page}: {url}")
        driver.get(url)
        time.sleep(3)
        # ---------- gradual scrolling ----------
        wait.until(
            EC.presence_of_all_elements_located(
                (
                    By.CSS_SELECTOR,
                    "div.MuiPaper-root.MuiPaper-outlined.MuiPaper-rounded",
                )
            )
        )

        scroll_pause = 2
        step = 600  # scroll 600px each step
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script(f"window.scrollBy(0, {step});")
            time.sleep(scroll_pause)
            new_height = driver.execute_script("return document.body.scrollHeight")
            # If we reached the bottom (height stops increasing)
            if new_height == last_height:
                break
            last_height = new_height
        # ---------- end ----------
        cards = driver.find_elements(
            By.CSS_SELECTOR, "div.MuiPaper-root.MuiPaper-outlined.MuiPaper-rounded"
        )

        for card in cards:
            try:
                title = card.find_element(By.XPATH, ".//a//p[2]").text
            except:
                title = "N/A"
            try:
                price = card.find_element(By.XPATH, './/p[contains(text(),"USD")]').text
            except:
                price = "N/A"
            try:
                Type = card.find_element(
                    By.XPATH,
                    ".//a//p[3]//span[1] | .//span[contains(@class, 'tj0ab2')]",
                ).text
            except:
                Type = "NA"
                print("Unable to get the Type ")
            try:
                bedroom = card.find_element(By.XPATH, ".//a//p[3]//span[2]").text
            except:
                bedroom = "N/A"
            try:
                bathroom = card.find_element(By.XPATH, ".//a//p[3]//span[3]").text
            except:
                bathroom = "N/A"
            try:
                location = card.find_element(
                    By.XPATH,
                    ".//p[contains(@class,'MuiTypography-body2') and not(./span)]",
                ).text
            except:
                location = "N/A"
            try:
                url = card.find_element(By.XPATH, ".//a ").get_attribute("href")
            except:
                url = "N/A"

            properties.append(
                {
                    "title": title,
                    "price": price,
                    "type": Type,
                    "bedrooms": bedroom,
                    "bathrooms": bathroom,
                    "location": location,
                    "url": url,
                }
            )

    driver.quit()
    return properties


def threading_detail_Parsing(links):
    driver = setup_driver()
    for link in links:

        driver.get(link)
        print(link)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # --------- AMENITIES ---------
        amenities = []
        try:
            amenities_header = soup.find("h4", string=lambda x: x and "Amenities" in x)
            if amenities_header:
                amenities_container = amenities_header.find_next_sibling("div")
                if amenities_container:
                    items = amenities_container.find_all(
                        "div", class_=lambda c: c and "MuiGrid-item" in c
                    )
                    for item in items:
                        txt = item.get_text(strip=True)
                        if txt:
                            txt = txt.replace("\u200b", "")
                            amenities.append(txt)

        except:
            print("Unable to get Amenities")

        # --------- DESCRIPTION ---------
        description = ""
        try:
            desc_header = soup.find("h4", string=lambda x: x and "Description" in x)
            if desc_header:
                paragraphs = []
                # get ALL p tags after Description
                for p in desc_header.find_all_next("p"):
                    # Stop when next section begins (next h4)
                    prev_h4 = p.find_previous("h4")
                    if prev_h4 != desc_header:
                        break
                    paragraphs.append(p.get_text(strip=True))

                description = " ".join(paragraphs).replace("\n", " ").strip()

        except:
            print("Unable to get Description")

        sqft_tag = soup.find(string=lambda x: x and "sqft" in x.lower())
        if sqft_tag:
            sqft = sqft_tag.split("/")[0].strip()
        else:
            sqft = "NA"

        all_Details.append(
            {
                "url": link,
                "description": description,
                "amenities": amenities,
                "area": sqft,
                "country": "Lebanon",
            }
        )
    driver.quit()


all_Details = []


def parallel_scraping_threading(Links):
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(threading_detail_Parsing, Links)


rows = scrape_features(pages=150)
df1 = pd.DataFrame(rows)
details = df1
details.to_csv("properties.csv", index=False)
Properties_Links = [item["url"] for item in rows if item["url"] != "N/A"]
# parallel_scraping_threading(Properties_Links)
threading_detail_Parsing(Properties_Links)
df1 = pd.DataFrame(rows)
df2 = pd.DataFrame(all_Details)
final = df1.merge(df2, on="url", how="left")
final.to_csv("detailed_properties.csv", index=False)
