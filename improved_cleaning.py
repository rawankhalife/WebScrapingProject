import pandas as pd
import numpy as np
import re
import ast
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def improved_data_cleaning(df, country_name):
    """
    Enhanced data cleaning with better outlier handling and data validation
    """
    print(f"Starting improved cleaning for {country_name}")
    print(f"Original shape: {df.shape}")

    df = df.copy()

    # ======================================================
    # 1. BASIC DATA TYPE CONVERSIONS AND CLEANING
    # ======================================================

    # Price cleaning with currency conversion
    if "price" in df.columns:
        df["price"] = df["price"].apply(extract_price_improved)

    # Area cleaning
    if "area" in df.columns:
        df["area"] = df["area"].apply(extract_area_improved)

    # Bedroom/Bathroom cleaning
    for col in ["bedrooms", "bathrooms"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_integer_improved)

    # Year built cleaning
    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].apply(extract_year_improved)
        # Calculate property age
        df["property_age"] = 2025 - df["year_built"]
        df["property_age"] = df["property_age"].clip(
            lower=0, upper=200
        )  # Reasonable bounds

    # ======================================================
    # 2. SMART OUTLIER DETECTION AND REMOVAL
    # ======================================================

    # Price outlier detection using statistical methods
    if "price" in df.columns:
        df = remove_price_outliers(df, country_name)

    # Area outlier detection
    if "area" in df.columns:
        df = remove_area_outliers(df)

    # Price per sqft validation
    if "price" in df.columns and "area" in df.columns:
        df = validate_price_per_sqft(df, country_name)

    # ======================================================
    # 3. MISSING VALUE IMPUTATION
    # ======================================================
    df = smart_missing_value_imputation(df)

    # ======================================================
    # 4. DATA VALIDATION AND CONSISTENCY CHECKS
    # ======================================================
    df = validate_data_consistency(df)

    print(f"Final shape after cleaning: {df.shape}")

    return df


def extract_price_improved(text):
    """Enhanced price extraction with currency conversion"""
    if pd.isna(text):
        return np.nan

    text_str = str(text).strip().upper()

    # Check for currency indicators
    is_iqd = any(indicator in text_str for indicator in ["IQD", "DINAR", "دينار"])
    is_lbp = any(indicator in text_str for indicator in ["LBP", "LL", "LEBANESE"])
    is_usd = any(indicator in text_str for indicator in ["USD", "$", "DOLLAR"])

    # Extract numeric value
    num_match = re.search(r"[\d,.-]+", text_str.replace(",", ""))
    if not num_match:
        return np.nan

    try:
        value = float(num_match.group())

        # Convert to USD based on currency
        if is_iqd:
            value = value / 1310  # IQD to USD conversion rate
        elif is_lbp:
            value = value / 15000  # LBP to USD conversion rate (approximate)
        # USD is already in correct currency

        return max(0, value)  # Ensure non-negative
    except:
        return np.nan


def extract_area_improved(text):
    """Enhanced area extraction with unit conversion"""
    if pd.isna(text):
        return np.nan

    text_str = str(text).strip().lower()

    # Extract numeric value
    num_match = re.search(r"[\d,.-]+", text_str.replace(",", ""))
    if not num_match:
        return np.nan

    try:
        value = float(num_match.group())

        # Convert different units to square feet
        if any(unit in text_str for unit in ["m²", "sqm", "square meter"]):
            value = value * 10.764  # Convert sqm to sqft
        elif any(unit in text_str for unit in ["acre", "acres"]):
            value = value * 43560  # Convert acres to sqft
        # Assume sqft if no unit specified

        return max(0, value)
    except:
        return np.nan


def extract_integer_improved(text):
    """Enhanced integer extraction"""
    if pd.isna(text):
        return np.nan

    if isinstance(text, (int, float)) and not np.isnan(text):
        return max(0, int(text))

    num_match = re.search(r"\d+", str(text))
    if num_match:
        return max(0, int(num_match.group()))

    return np.nan


def extract_year_improved(text):
    """Enhanced year extraction with validation"""
    if pd.isna(text):
        return np.nan

    # Look for 4-digit year between 1800 and 2025
    year_match = re.search(r"\b(18|19|20)\d{2}\b", str(text))
    if year_match:
        year = int(year_match.group())
        if 1800 <= year <= 2025:
            return year

    return np.nan


def remove_price_outliers(df, country_name):
    """Country-specific price outlier removal"""
    if "price" not in df.columns:
        return df

    original_count = len(df)

    # Country-specific price bounds based on market knowledge
    country_bounds = {
        "IRAQ": {"min": 5000, "max": 3_000_000},
        "LEBANON": {"min": 15000, "max": 8_000_000},
        "USA": {"min": 10000, "max": 20_000_000},
    }

    bounds = country_bounds.get(country_name.upper(), {"min": 5000, "max": 10_000_000})

    # Apply bounds
    df = df[(df["price"] >= bounds["min"]) & (df["price"] <= bounds["max"])]

    # Additional statistical outlier removal using modified Z-score
    price_median = df["price"].median()
    mad = np.median(np.abs(df["price"] - price_median))  # Median Absolute Deviation

    if mad > 0:
        modified_z_scores = 0.6745 * (df["price"] - price_median) / mad
        df = df[np.abs(modified_z_scores) < 3.5]  # Keep data within 3.5 MAD

    removed_count = original_count - len(df)
    print(
        f"Removed {removed_count} price outliers ({removed_count/original_count*100:.1f}%)"
    )

    return df


def remove_area_outliers(df):
    """Remove unrealistic area values"""
    if "area" not in df.columns:
        return df

    original_count = len(df)

    # Reasonable area bounds (50 sqft to 50,000 sqft)
    df = df[(df["area"] >= 50) & (df["area"] <= 50_000)]

    # Additional statistical check
    q1, q99 = df["area"].quantile([0.01, 0.99])
    df = df[(df["area"] >= q1) & (df["area"] <= q99)]

    removed_count = original_count - len(df)
    if removed_count > 0:
        print(
            f"Removed {removed_count} area outliers ({removed_count/original_count*100:.1f}%)"
        )

    return df


def validate_price_per_sqft(df, country_name):
    """Validate price per square foot ratios"""
    if "price" not in df.columns or "area" not in df.columns:
        return df

    # Calculate price per sqft
    df["price_per_sqft"] = df["price"] / df["area"]

    # Country-specific reasonable price per sqft bounds
    bounds = {
        "IRAQ": {"min": 10, "max": 500},
        "LEBANON": {"min": 20, "max": 800},
        "USA": {"min": 30, "max": 1500},
    }

    country_bounds = bounds.get(country_name.upper(), {"min": 10, "max": 1000})

    original_count = len(df)
    df = df[
        (df["price_per_sqft"] >= country_bounds["min"])
        & (df["price_per_sqft"] <= country_bounds["max"])
    ]

    removed_count = original_count - len(df)
    if removed_count > 0:
        print(
            f"Removed {removed_count} properties with unrealistic price/sqft ({removed_count/original_count*100:.1f}%)"
        )

    return df


def smart_missing_value_imputation(df):
    """Intelligent missing value imputation"""

    # For numeric columns, use median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # For categorical columns, use mode or "Unknown"
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            fill_value = mode_value[0] if len(mode_value) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def validate_data_consistency(df):
    """Check and fix data consistency issues"""

    # Ensure bedrooms and bathrooms are non-negative
    for col in ["bedrooms", "bathrooms"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=50)  # Reasonable upper bound

    # Ensure property age is reasonable
    if "property_age" in df.columns:
        df["property_age"] = df["property_age"].clip(lower=0, upper=200)

    # Fix impossible combinations (e.g., studio with 5+ bedrooms)
    if "bedrooms" in df.columns and "property_type" in df.columns:
        studio_mask = df["property_type"].str.contains(
            "studio|Studio", case=False, na=False
        )
        df.loc[studio_mask, "bedrooms"] = df.loc[studio_mask, "bedrooms"].clip(upper=1)

    return df


def clean_text_features(df):
    """Clean and process text features"""

    text_columns = ["description", "title", "amenities"]

    for col in text_columns:
        if col in df.columns:
            # Basic text cleaning
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", "None", ""], np.nan)

            # Create word count features
            df[f"{col}_word_count"] = df[col].str.split().str.len()
            df[f"{col}_char_count"] = df[col].str.len()

    return df


def process_amenities_improved(df):
    """Enhanced amenity processing"""

    if "amenities" not in df.columns:
        return df

    # Define amenity categories
    amenity_categories = {
        "security": ["security", "guard", "gated", "alarm", "cctv", "surveillance"],
        "parking": ["parking", "garage", "carport"],
        "pool": ["pool", "swimming", "jacuzzi", "spa"],
        "gym": ["gym", "fitness", "exercise", "workout"],
        "kitchen": ["kitchen", "cooking", "appliances", "dishwasher"],
        "outdoor": ["garden", "balcony", "terrace", "patio", "yard"],
        "luxury": ["marble", "granite", "hardwood", "luxury", "premium"],
        "utilities": ["electricity", "water", "gas", "internet", "cable"],
    }

    # Count amenities by category
    for category, keywords in amenity_categories.items():
        df[f"has_{category}_amenities"] = df["amenities"].apply(
            lambda x: count_amenity_matches(x, keywords)
        )

    # Total amenity count
    df["total_amenities"] = df["amenities"].apply(count_total_amenities)

    return df


def count_amenity_matches(amenity_text, keywords):
    """Count matches for amenity keywords"""
    if pd.isna(amenity_text):
        return 0

    text_lower = str(amenity_text).lower()
    return sum(1 for keyword in keywords if keyword in text_lower)


def count_total_amenities(amenity_text):
    """Count total number of amenities"""
    if pd.isna(amenity_text):
        return 0

    try:
        # Try to parse as list
        amenities = ast.literal_eval(str(amenity_text))
        if isinstance(amenities, list):
            return len(amenities)
    except:
        pass

    # Count by splitting on common delimiters
    delimiters = [",", ";", "\n", "|"]
    for delim in delimiters:
        if delim in str(amenity_text):
            return len(str(amenity_text).split(delim))

    return 1 if str(amenity_text).strip() else 0


def main_cleaning_pipeline(country_name):
    """Main cleaning pipeline for a country"""

    print(f"\n{'='*60}")
    print(f"IMPROVED CLEANING PIPELINE FOR {country_name.upper()}")
    print(f"{'='*60}")

    # Read data
    file_path = f"{country_name}/{country_name.title()}_Listings.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        pass
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    # Apply improved cleaning
    df_cleaned = improved_data_cleaning(df, country_name)

    # Process text features
    df_cleaned = clean_text_features(df_cleaned)

    # Process amenities
    df_cleaned = process_amenities_improved(df_cleaned)

    # Save cleaned data
    output_path = f"{country_name}/{country_name}_improved_cleaned_listings.csv"
    df_cleaned.to_csv(output_path, index=False)

    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final record count: {len(df_cleaned)}")
    print(f"Data reduction: {(len(df) - len(df_cleaned))/len(df)*100:.1f}%")

    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    if "price" in df_cleaned.columns:
        print(
            f"Price range: ${df_cleaned['price'].min():,.0f} - ${df_cleaned['price'].max():,.0f}"
        )
        print(f"Price median: ${df_cleaned['price'].median():,.0f}")

    if "area" in df_cleaned.columns:
        print(
            f"Area range: {df_cleaned['area'].min():.0f} - {df_cleaned['area'].max():.0f} sqft"
        )

    return df_cleaned


def predicting_cleaning(df, country_name):

    # Apply improved cleaning
    df_cleaned = improved_data_cleaning(df, country_name)

    # Process text features
    df_cleaned = clean_text_features(df_cleaned)

    # Process amenities
    df_cleaned = process_amenities_improved(df_cleaned)

    if "area" in df_cleaned.columns:
        print(
            f"Area range: {df_cleaned['area'].min():.0f} - {df_cleaned['area'].max():.0f} sqft"
        )

    return df_cleaned


if __name__ == "__main__":
    countries = ["iraq"]

    for country in countries:
        try:
            cleaned_df = main_cleaning_pipeline(country)
        except Exception as e:
            print(f"Error cleaning {country}: {e}")
            import traceback

            traceback.print_exc()
