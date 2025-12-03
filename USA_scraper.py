import os
import time
import json
import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

start_page = 12
end_page = 32

from selenium.webdriver.common.by import By


def scrape_price_history(driver):
    """
    Scrapes Redfin price history (Sale History tab).
    Returns list of dicts: [{date, event, price, source}, ...]
    """

    history_events = []

    # Locate all history rows
    rows = driver.find_elements(
        By.XPATH, "//div[contains(@class,'PropertyHistoryEventRow')]"
    )

    for row in rows:
        # DATE
        try:
            date_text = row.find_element(By.XPATH, ".//div[@class='col-4']/p[1]").text
        except:
            date_text = None

        # EVENT TYPE
        try:
            event_text = row.find_element(
                By.XPATH, ".//div[contains(@class,'description-col')]/div[1]"
            ).text
        except:
            event_text = None

        # PRICE
        try:
            price_text = row.find_element(
                By.XPATH, ".//div[contains(@class,'price-col')]"
            ).text

            # Clean
            price_text = price_text.split("(")[0].strip()
            if price_text in ["—", "*", ""]:
                price_value = None
            else:
                price_value = int(price_text.replace("$", "").replace(",", "").strip())
        except:
            price_value = None

        # MLS or source info (optional)
        try:
            source_text = row.find_element(
                By.XPATH,
                ".//div[contains(@class,'description-col')]/p[@class='subtext']",
            ).text
        except:
            source_text = None

        # Append dict
        history_events.append(
            {
                "date": date_text,
                "event": event_text,
                "price": price_value,
                "source": source_text,
            }
        )

    return history_events


###############################################
# 1. SELENIUM SETUP (same as your working code)
###############################################
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )


###############################################
# 2. SCRAPE LISTING URLs FROM A CITY (Selenium)
###############################################
def scrape_city_urls(city_url):
    print(f"\n🌆 Scraping listing URLs + basic info from: {city_url}")
    driver = setup_driver()
    driver.get(city_url)

    time.sleep(4)

    listings = []
    seen_urls = set()
    prev_count = 0

    while True:

        # Scroll to load listings
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.2, 2.0))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        cards = soup.find_all("div", class_="HomeCardContainer")

        for home in cards:
            # URL
            a = home.find("a", class_="bp-Homecard__Address")
            if not a:
                continue

            url = "https://www.redfin.com" + a["href"]

            if url in seen_urls:
                continue
            seen_urls.add(url)

            # ADDRESS
            address = a.get_text(strip=True) if a else None

            # PRICE
            price_tag = home.select_one(".bp-Homecard__Price--value")
            price = price_tag.get_text(strip=True) if price_tag else None

            # BEDS
            beds_tag = home.select_one(".bp-Homecard__Stats--beds")
            beds = beds_tag.get_text(strip=True) if beds_tag else None

            # BATHS
            baths_tag = home.select_one(".bp-Homecard__Stats--baths")
            baths = baths_tag.get_text(strip=True) if baths_tag else None

            # SQFT
            sqft_tag = home.select_one(".bp-Homecard__Stats--sqft")
            sqft = sqft_tag.get_text(" ", strip=True) if sqft_tag else None
            if sqft and "—" in sqft:
                sqft = None

            # -----------------------------
            # Save row
            # -----------------------------
            listings.append(
                {
                    "url": url,
                    "address": address,
                    "price": price,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "area": sqft,
                }
            )

        # Stop if nothing new is found
        if len(seen_urls) == prev_count:
            print("No new listings — stopping pagination.")
            break

        prev_count = len(seen_urls)

        # NEXT button
        next_btn = driver.find_elements("css selector", "button[aria-label='next']")
        if not next_btn:
            print("No NEXT button — final page reached.")
            break

        btn = next_btn[0]
        print("   ➡ Clicking NEXT page...")

        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", btn
            )
            time.sleep(1)
            btn.click()
        except:
            driver.execute_script("arguments[0].click();", btn)

        time.sleep(random.uniform(2.5, 4.0))

    driver.quit()
    print(f"\n✔ TOTAL listings collected: {len(listings)}")
    return listings


###############################################
# 3. THREAD-SAFE DETAIL SCRAPER (Requests)
###############################################
def scrape_detail(url, driver):
    try:
        driver.get(url)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # ---------------------------------
        # DESCRIPTION
        # ---------------------------------
        desc_tag = soup.select_one("div.remarks p")
        description = desc_tag.get_text(" ", strip=True) if desc_tag else None

        # ---------------------------------
        # LUXURY TAGS
        # ---------------------------------
        tags = soup.select("span.highlightedTag")
        highlighted_features = [t.get_text(strip=True) for t in tags]

        # ---------------------------------
        # PROPERTY TYPE
        # ---------------------------------
        property_type = None
        for row in soup.select("div.keyDetails-row"):
            label = row.select_one("span.valueType")
            if label and "Property Type" in label.get_text(strip=True):
                property_type = row.select_one("span.valueText").get_text(strip=True)
                break

        # ---------------------------------
        # YEAR BUILT
        # ---------------------------------
        year_built = None
        for row in soup.select("div.keyDetails-row"):
            label = row.select_one("span.valueType")
            if label and "Year Built" in label.get_text(strip=True):
                year_built = row.select_one("span.valueText").get_text(strip=True)
                break

        # ---------------------------------
        # LOT SIZE
        # ---------------------------------
        lot_size = None
        lot_row = soup.find("li", text=lambda x: x and "Lot Size" in x)
        if lot_row:
            lot_size = lot_row.parent.find_all("li", class_="entryItem")[0].get_text(
                strip=True
            )

        # ---------------------------------
        # PARKING SECTION
        # ---------------------------------
        # Extract ALL entries under the Parking H3 section
        parking = []
        parking_section = soup.find("h3", text="Parking")
        if parking_section:
            ul = parking_section.find_next("ul")
            if ul:
                for li in ul.find_all("li", class_="entryItem"):
                    parking.append(li.get_text(strip=True))

        amenities = set()  # use a set to avoid duplicates

        # 1) UTILITIES (Sewer, Water Source, Gas, Electricity, Internet…)
        utilities_section = soup.find("h3", string=lambda t: t and "Utilities" in t)
        if utilities_section:
            util_block = utilities_section.find_next(
                "div", class_="super-group-content"
            )

            if util_block:
                # Grab items like "Sewer:" → keep only "Sewer"
                for li in util_block.find_all("li", class_="entryItem"):
                    text = li.get_text(strip=True)
                    if ":" in text:
                        category = text.split(":")[0].strip()
                        amenities.add(category)

                # Grab Electricity / Internet titles
                for title in util_block.find_all(
                    "div", class_="utilities-content-item-title"
                ):
                    name = title.get_text(" ", strip=True)

                    # Electricity
                    if "Electricity" in name:
                        amenities.add("Electricity")

                    # Internet
                    if "Internet" in name:
                        amenities.add("Internet")

        # 2) PARKING
        parking_section = soup.find("h3", string=lambda t: t and "Parking" in t)
        if parking_section:
            amenities.add("Parking")

        # 3) INTERIOR FEATURES
        interior_section = soup.find("h3", string=lambda t: t and "Interior" in t)
        if interior_section:
            amenities.add("Interior Features")

        # 4) EXTERIOR FEATURES
        exterior_section = soup.find("h3", string=lambda t: t and "Exterior" in t)
        if exterior_section:
            amenities.add("Exterior Features")

        # 5) HEATING / COOLING
        hvac_section = soup.find(
            "h3", string=lambda t: t and "Heating" in t or "Cooling" in t
        )
        if hvac_section:
            amenities.add("HVAC")

        # 6) POOL
        pool_li = soup.find("li", string=lambda t: t and "Pool" in t)
        if pool_li:
            amenities.add("Pool")

        # 7) FIREPLACE
        fireplace_li = soup.find("li", string=lambda t: t and "Fireplace" in t)
        if fireplace_li:
            amenities.add("Fireplace")

        # Convert set → list
        amenities = sorted(list(amenities))

        # PRICE TRENDS
        price_history = scrape_price_history(driver)

        # details
        sections = driver.find_elements(By.XPATH, "//h3[@class='super-group-title']")

        details = {}

        for section in sections:
            section_name = section.text.strip()

            # Find the content div after the h3
            content_div = section.find_element(
                By.XPATH, "./following-sibling::div[@class='super-group-content'][1]"
            )

            # All <ul> groups inside this section
            groups = content_div.find_elements(
                By.XPATH,
                ".//ul[@class='bulletList' or @class='bulletList no-break-inside']",
            )

            section_dict = {}

            for group in groups:
                # First <li> is the header
                header = group.find_element(
                    By.XPATH, ".//li[@class='propertyDetailsHeader']"
                ).text.strip()

                # The rest are entries
                entries = group.find_elements(By.XPATH, ".//li[@class='entryItem']")
                entry_values = [e.text.strip() for e in entries]

                section_dict[header] = entry_values

            details[section_name] = section_dict

        print(details)

        # ---------------------------------
        # RETURN EVERYTHING
        # ---------------------------------
        return {
            "description": description,
            "property_type": property_type,
            "year_built": year_built,
            "lot_size": lot_size,
            "parking": parking,
            "amenities": amenities,
            "highlighted_features": highlighted_features,
            "price_history": price_history,
            "details": details,
        }

    except Exception as e:
        return {"url": url, "error": str(e)}


from concurrent.futures import ThreadPoolExecutor, as_completed


def scrape_single_url(url):
    driver = setup_driver()  # each thread gets its own Chrome
    try:
        data = scrape_detail(url, driver)
    except Exception as e:
        data = {"url": url, "error": str(e)}
    driver.quit()
    return data


def scrape_all_details(urls, max_threads=4):
    print(f"Scraping {len(urls)} detail pages with {max_threads} threads...")

    results = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(scrape_single_url, url): url for url in urls}

        for future in as_completed(futures):
            url = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"url": url, "error": str(e)}

            results.append(res)

    return results


###############################################
# 4. CITY LIST TO SCRAPE  (simple for now)
###############################################
CITIES = {
    "CA": [
        {"id": 11203, "name": "Los-Angeles"},
        {"id": 16904, "name": "San-Diego"},
    ],
    "FL": [
        {"id": 11458, "name": "Miami"},
        {"id": 18142, "name": "Tampa"},
    ],
    "NY": [
        {"id": 30749, "name": "New-York"},
    ],
    "IL": [{"id": 29470, "name": "Chicago"}],
    "DC": [{"id": 12839, "name": "Washington-DC"}],
    "NV": [{"id": 10201, "name": "Las-Vegas"}],
    "WA": [{"id": 16163, "name": "Seattle"}],
    "CO": [{"id": 5155, "name": "Denver"}],
    "TN": [{"id": 13415, "name": "Nashville"}],
    "MO": [{"id": 35751, "name": "Kansas-City"}],
}


###############################################
# 5. MAIN MULTI-STATE RUNNER
###############################################
def run_all():
    os.makedirs("redfin_data", exist_ok=True)
    all_rows = []

    for state, citylist in CITIES.items():
        print(f"\n==============================")
        print(f"STATE: {state}")
        print(f"==============================")

        state_rows = []

        for city in citylist:
            city_id = city["id"]
            name = city["name"]
            url = f"https://www.redfin.com/city/{city_id}/{state}/{name}"

            print(f"\n Scraping city: {name} ({state})")

            # 1) Collect listing URLs via Selenium
            basic_rows = scrape_city_urls(url)
            urls = [row["url"] for row in basic_rows]
            urls = urls[:1]
            # 2) Scrape details in parallel
            details = scrape_all_details(urls)

            # merge basic + detail
            merged = []
            for base, det in zip(basic_rows, details):
                merged.append({**base, **det})

            # Add city/state metadata
            for d in merged:
                d.pop("error", None)
                d["state"] = state
                d["city"] = name
                d["country"] = "USA"
                state_rows.append(d)
                all_rows.append(d)

        # Save state file
        df = pd.DataFrame(state_rows)
        output_path = f"redfin_data/redfin_{state}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved → {output_path}")

    # Save master file
    pd.DataFrame(all_rows).to_csv("redfin_data/redfin_master.csv", index=False)
    print("\n ALL DONE — Saved master dataset in redfin_data/redfin_master.csv")


###############################################
# RUN
###############################################
if __name__ == "__main__":
    run_all()
