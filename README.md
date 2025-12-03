# 🏠 Real Estate Web Scraping, Analysis & Price Prediction  
A multi-country real estate analytics project covering **Lebanon**, **Iraq**, and the **USA**.  
We built full web scrapers, cleaned and standardized the data, engineered new features, trained machine-learning models, and deployed an interactive Streamlit dashboard for predictions and market insights.

---

## 📌 Project Overview  
The goal of this project is to collect real estate listings from different countries, clean and unify the datasets, and build a machine-learning model that predicts property prices based on area, bedrooms, bathrooms, property type, amenities, and descriptions.  
The dashboard allows users to enter property details and instantly receive a predicted price along with market insights.

---

## 📂 Repository Structure  

├── EDA/ # Exploratory Data Analysis notebooks
├── iraq/ # Iraq data + trained models
├── lebanon/ # Lebanon data + trained models
├── usa/ # USA data + trained models
│
├── Iraq_scraper.py # Iraq web scraper
├── Lebanon_scrapper.py # Lebanon web scraper
├── USA_scraper.py # USA web scraper
│
├── improved_cleaning.py # Cleaning & preprocessing functions
├── feature_engineering.py # Feature engineering pipeline
├── improved_models.py # Model training scripts
├── dashboard.py # Streamlit dashboard app
│
├── requirements.txt # Dependencies
└── README.md


---


## Web Scraping

Each country has its own scraping script due to differences in website structure, layout, and available data.

### Lebanon
- Scrapes: realestate.com.lb  
- Uses Selenium for infinite scrolling and dynamic content  
- Extracts: title, price, type, bedrooms, bathrooms, location  
- A second function processes detail pages for descriptions and amenities  

### Iraq
- Scrapes: homele.com  
- Extracts listing card data (title, bedrooms, bathrooms, area, location)  
- Converts area from m² to sqft  
- Uses BeautifulSoup to parse detail pages  

### USA
- Scrapes: Redfin  
- Selenium for pagination and URL extraction  
- Multi-threaded detail scraping  
- Extracts detailed metadata: price history, amenities, year built, lot size, etc.

All scrapers:
- Respect robots.txt  
- Use short delays and controlled scrolling  
- Only collect publicly visible information  

---

## Data Cleaning

Located in `improved_cleaning.py`.

Main steps:
- Converting strings to numerical values
- Fixing inconsistent area/price formats
- Normalizing city and location fields
- Cleaning room counts
- Handling missing values
- Removing duplicates
- Cleaning and standardizing the "amenities" and "description" fields

---

## Feature Engineering

Implemented in `feature_engineering.py`.

Includes:
- Word counts (description, title)
- Property age
- Amenity one-hot encoding
- Type and city one-hot encoding
- Interaction features (area × bedrooms, bedrooms × bathrooms)
- Polynomial features (area², log(area))
- Text keyword flags (e.g., “view”, “renovated”, “modern”)
- Structural ratios:  
  - area per room  
  - room density  
  - bed/bath ratios

The goal is to transform the raw scraped data into consistent numerical inputs for the models.

---

## Machine Learning Models

Trained separately for each country using:
- XGBoost
- Polynomial and interaction features
- Robust scaling for numeric stability
- Cross-validation for evaluation

Model performance:
- **Iraq:** ~82% R²  
- **Lebanon:** ~75% R²  
- **USA:** ~58% R² (more complex and varied market)

Training code is in `improved_models.py`.

---

## Dashboard (Streamlit)

`dashboard.py` provides:

### Price Predictor
- User enters property details (country, city, type, area, rooms, amenities)
- Model predicts price
- Shows price per sqft, category, and a breakdown chart

### Market Analytics
- Price by city  
- Price by property type  
- Bedroom distributions  
- Correlation heatmaps  
- Area vs price scatter  
- Interactive filters  

### NLP Search
- Searches listings by keywords inside descriptions, titles, and locations
