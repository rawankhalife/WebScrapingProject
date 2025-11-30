import nltk
import re
import ast
import numpy as np
import pandas as pd
from scipy import stats
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')


def extract_number(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r"[\d,.]+", str(text))
    if not m:
        return np.nan
    
    num_str = m.group().replace(",", "")
    
    # If it's a float like "1722.22", convert to float then to int
    if "." in num_str:
        return int(float(num_str))  
    return int(num_str)

def iqr_bounds(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return lower, upper

def extract_integer(text):
    if pd.isna(text):
        return np.nan
    
    if isinstance(text, int):
        return text
    
    if isinstance(text, float):
        return int(text)
    
    text = str(text)
    
    m = re.search(r"\d+(\.\d+)?", text)
    if not m:
        return np.nan
    
    return int(float(m.group()))


def clean_description(text):
    if pd.isna(text) or text.strip() == "":
        return "NA"
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    return re.sub(r"\s+", " ", text).strip()

def clean_lot_size(val):
    if pd.isna(val):
        return np.nan
    val = str(val)
    sq = re.search(r"Lot Size \(Sq\. Ft\.\):\s*([\d,]+)", val)
    if sq:
        return int(sq.group(1).replace(",", ""))
    ac = re.search(r"Lot Size \(Acres\):\s*([\d.]+)", val)
    if ac:
        return float(ac.group(1)) * 43560
    return np.nan

def extract_year(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r"\b(18|19|20)\d{2}\b", str(text))
    return int(m.group()) if m else np.nan

# ---------------------------
# NLP helpers
# ---------------------------

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def clean_basic(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_stop(text):
    return " ".join(t for t in text.split() if t not in STOPWORDS)

def lemma(text):
    return " ".join(lemmatizer.lemmatize(t) for t in text.split())

# def preprocess_text(text):
#     text = clean_basic(text)
#     text = remove_stop(text)
#     text = lemma(text)
#     return text
def preprocess_text(text):
    if pd.isna(text):
        return text   # keep NaN as-is
    text = clean_basic(text)
    text = remove_stop(text)
    text = lemma(text)
    return text

def split_location(text, country):
    parts = [p.strip() for p in str(text).split(",")]

    # -------------------------------------------
    # IRAQ (example: "Baghlumnara, Erbil")
    # -------------------------------------------
    if country == "USA":
        pass

    if country == "IRAQ":
        if len(parts) >= 2:
            return pd.Series({
                "city": parts[1],
            })
        else:
            return pd.Series({"city": parts[1]})

    # -------------------------------------------
    # LEBANON (example: "Zalka, Mount Lebanon Governorate, El Metn district")
    # -------------------------------------------
    if country == "LEBANON":
        if len(parts) == 3:
            return pd.Series({
                "city": parts[0],
                "district": parts[2].replace("district", "").replace("districts", "")
            })
        elif len(parts) == 2:
            return pd.Series({
                "city": parts[0],
                "district": "NA"
            })
        else:
            return pd.Series({"city": parts[0], "district": "NA"})
        

    # -------------------------------------------
    # USA or anything else → no custom split
    # -------------------------------------------
    return pd.Series({"city": "NA", "district": "NA"})


type_map = {
    'Apartment': 'apartment','Condo': 'apartment','Condo (co-op)': 'apartment','Loft': 'apartment',
    'Penthouse': 'penthouse',
    'House':'single_house','Single-family':'single_house',
    'Single Family Residence, 24 - Floating Home/On-Water Res':'single_house',
    'Villa':'villa',
    'Townhome':'townhouse','Townhouse':'townhouse','Townhome (co-op)':'townhouse',
    'Duplex':'duplex','Triplex':'triplex','Multi-family':'multi_family',
    'Manufactured':'manufactured','Moorage':'moorage',
    'Vacant land':'residential_land','Land':'residential_land','Residential Plot':'residential_land',
    'Commercial Plot':'commercial_land','Industrial Land':'industrial_land',
    'Shop':'retail','Showroom':'retail','Restaurant':'retail','Beauty Salon':'retail',
    'Office':'office','Warehouse':'warehouse',
    'Hotel':'hotel','Hotel Apartments':'hotel',
    'Commercial Building':'commercial_building','Commercial Property':'commercial_building',
    'Commercial House':'commercial_building','Commercial Floor':'commercial_building',
    'Full floor':'commercial_building','Half floor':'commercial_building','Whole Building':'commercial_building',
    'Farm':'farm_chalet','Chalet':'farm_chalet',
    'Parking':'parking_unit'
}


def clean_data(df):
    df = df.copy()

    if "url" in df.columns:
        df.drop_duplicates(subset=["url"], inplace=True)

    if "description" in df.columns:
        df["description"] = df["description"].apply(clean_description)

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.replace("-", " ").str.title()

    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper()

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.upper()

    for col, func in [
        ("price", extract_number),
        ("bedrooms", extract_integer),
        ("bathrooms", extract_integer),
        ("area", extract_number),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(func)


    if "bedrooms" in df.columns:
        df["bedrooms"] = df["bedrooms"].replace(
            {"studio": 1, "Studio": 1, "STUDIO": 1}
        )
        df["bedrooms"] = df["bedrooms"].apply(
            lambda x: 1 if str(x).strip().lower() == "studio" else x
        )

    # --- lot_size ---
    if "lot_size" in df.columns:
        df["lot_size"] = df["lot_size"].apply(clean_lot_size)

    # --- year_built ---
    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].apply(extract_year)

    # ---------------------------
    # AMENITIES CLEANING
    # ---------------------------
    amenity_groups = {
        "pool": ["Pool","Swimming Pool","Aqua Park","Private Pool","Shared Pool","Children's Pool","Private Jacuzzi"],
        "fireplace": ["Fireplace"],
        "parking": ["Parking","Parking Spaces","Covered Parking"],
        "kitchen": ["Furnished Kitchen","Kitchen Counter","Separate kitchen","Kitchen Appliances","Pantry"],
        "air_conditioning": ["Air Conditioners Installed","Air Cooler","Central Air-Conditioner","Central A/C"],
        "security": ["CCTV Security","Security Staff","Fire Fighting","Security","Concierge"],
        "sports_recreation": ["Gym","Sport Center","Playground","Sauna","Shared Gym","Private Gym","Shared Spa"],
        "water_supply": ["24hr Water","Water Source","Water Well","Water Tank"],
        "electricity": ["24hr Electricity","Central Gas","Solar Power System","Boiler"],
        "balcony": ["Balcony","Terrace"],
        "basement": ["Basement"],
        "library": ["Library"],
        "mosque": ["Mosque"],
        "school": ["School"],
        "maintenance": ["Maintenance","Maid Service"],
        "office": ["Office","Study","Conference room"],
        "restaurant": ["Restaurant","Near restaurants"],
        "reception": ["Reception","Lobby in Building"],
        "children": ["Children's Play Area"],
        "garden": ["Private Garden"],
        "views": ["City View","View of Landmark","View of Water"],
        "pets": ["Pets Allowed"],
        "storage": ["Storage Room"],
        "maids_room": ["Maids Room"],
        "wardrobes": ["Built in Wardrobes","Walk-in Closet"],
        "transport": ["Near Public transportation"],
        "networking": ["Networked"],
        "barbecue": ["Barbecue Area"]
    }

    if "amenities" in df.columns:

        def group_amenities(cell):
            if pd.isna(cell) or cell.strip() == "":
                return []
            try:
                items = ast.literal_eval(cell)
            except:
                items = []
            matched = []
            for group, originals in amenity_groups.items():
                if any(a in items for a in originals):
                    matched.append(group)
            return matched

        df["amenities_old"] = df["amenities"]
        df["amenities"] = df["amenities"].apply(group_amenities)

    # PROPERTY TYPES
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].map(type_map).fillna("other")

    # NLP fields
    if "description" in df.columns:
        df["desc_nlp"] = df["description"].apply(preprocess_text)
        df["desc_nlp"] = df["desc_nlp"].fillna("Unknown")
        df["desc_nlp"] = df["desc_nlp"].astype(str)
        df["desc_nlp"] = df["desc_nlp"].replace(["na","NA","nan","NaN"], "Unknown")

    if "title" in df.columns:
        df["title_nlp"] = df["title"].apply(preprocess_text)

    # -------------------------------------------
    # COUNTRY-SPECIFIC LOCATION EXTRACTION
    # -------------------------------------------
    if "location" in df.columns and "country" in df.columns:
        location_expanded = df.apply(
            lambda row: split_location(row["location"], row["country"]),
            axis=1
        )
        df = pd.concat([df, location_expanded], axis=1)

    # -------------------------------------------
    # MISSING VALUES
    # -------------------------------------------
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("NA")

    return df

df = pd.read_csv("lebanon/Lebanon_Listings.csv")
clean_df = clean_data(df)


obj_cols = clean_df.select_dtypes(include="object").columns
clean_df[obj_cols] = clean_df[obj_cols].fillna("NA")


numeric_cols = clean_df.select_dtypes(include=["float", "int"]).columns
clean_df[numeric_cols] = clean_df[numeric_cols].fillna("NA")  

clean_df.head()
clean_df.to_csv("lebanon/lebanon_new_cleaned_listings.csv", index=False)