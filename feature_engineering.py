import re
import numpy as np
import pandas as pd
import ast


def add_features(df):
    df = df.copy()

    if "description" in df.columns:
        df["description_word_count"] = (
            df["description"].astype(str).apply(lambda x: len(x.split()))
        )

    if "title" in df.columns:
        df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))

    if "year_built" in df.columns:
        df["property_age"] = 2025 - df["year_built"]


    if "property_type" in df.columns:
        dummies = pd.get_dummies(df["property_type"], prefix="ptype", dtype=int)
        df = pd.concat([df, dummies], axis=1)


    if "amenities" in df.columns:
        df["num_amenities"] = df["amenities"].apply(len)

        for cell in df["amenities"]:
            if pd.isna(cell):  
                continue
            try:
                amenities_list = ast.literal_eval(cell)
                unique_amenities.update(amenities_list)
            except Exception as e:
                print(f"Skipping invalid row: {cell} ({e})")

        for group in unique_amenities:
            df[f"amenity_{group}"] = df["amenities"].apply(lambda x: int(group in x))

    Lebanon_KeyFeastures = [
        "company",
        "equiped",
        "global",
        "international",
        "concierge",
        "dining",
        "exceptional",
        "floor",
        "insurance",
        "master",
        "modern",
        "location",
        "investment",
        "marketing",
        "save",
        "world",
        "spacious",
        "offer",
    ]

    usa_KeyFeastures = [
        "additional",
        "dining",
        "comfort",
        "floor",
        "kitchen",
        "large",
        "located",
        "modern",
        "new",
        "offer",
        "perfect",
        "private",
        "spacious",
        "suite",
        "unit",
    ]

    iraq_KeyFeastures = [
        "offer",
        "plan",
        "tourism",
        "commercial",
        "complex",
        "furnished",
        "building",
        "city",
        "floor",
        "hall",
        "kitchen",
        "located",
        "negotiable",
        "payment",
        "residential",
        "sale",
        "tower",
        "village",
    ]

    country = df["country"][0]

    if country == "USA":
        for kw in usa_KeyFeastures:
            df[f"kw_{kw.replace(' ', '_')}"] = (
                df["desc_nlp"].str.contains(kw, case=False, na=False).astype(float)
            )
    elif country == "LEBANON":
        for kw in Lebanon_KeyFeastures:
            df[f"kw_{kw.replace(' ', '_')}"] = (
                df["desc_nlp"].str.contains(kw, case=False, na=False).astype(float)
            )
    else:
        for kw in iraq_KeyFeastures:
            df[f"kw_{kw.replace(' ', '_')}"] = (
                df["desc_nlp"].str.contains(kw, case=False, na=False).astype(float)
            )

    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    df["is_large_property"] = (df["area"] > df["area"].median()).astype(int)
    # df["is_large_property"] = (df["area"] > df["area"].quantile(0.75)).astype(int)


    lux = [
        "Private Jacuzzi",
        "Barbecue area",
        "Walk-in closet",
        "Conference Room",
        "Shared Spa",
        "Private Garden",
        "Jacuzzi",
        "Sauna",
        "Turkish Bath",
    ]

    if country == "LEBANON":
        df["has_luxury"] = df["amenities_old"].apply(
            lambda x: int(any(a in x for a in lux))
        )

    elif country == "USA":
        df["is_new_property"] = (
            df["property_age"] < df["property_age"].median()
        ).astype(int)
    else:
        pass

    # -----------------------------------
    # 6) LABEL ENCODING INTO NEW COLUMNS
    # -----------------------------------
    from sklearn.preprocessing import LabelEncoder

    cat_cols = ["city", "district", "state"]

    for col in cat_cols:
        if col in df.columns:
            # Create name for the new encoded column
            new_col = col + "_LE"

            # Convert to string and handle missing
            temp = df[col].fillna("Unknown").astype(str)

            le = LabelEncoder()
            df[new_col] = le.fit_transform(temp)

    # ----------------------------------------
    # 6.1 STRUCTURAL RATIO FEATURES
    # ----------------------------------------
    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["area_per_room"] = df["area"] / (df["total_rooms"].replace(0, np.nan))
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)

    if "bedrooms" in df.columns:
        df["is_family_sized"] = (df["bedrooms"] >= 3).astype(int)
        df["is_studio_flag"] = ((df["bedrooms"] <= 1) & (df["area"] < df["area"].median())).astype(int)


    # ----------------------------------------
    # 6.3 NLP-BASED QUALITY SIGNALS
    # ----------------------------------------
    if "desc_nlp" in df.columns:
        df["has_view"] = df["desc_nlp"].str.contains("view", case=False, na=False).astype(int)
        df["is_furnished"] = df["desc_nlp"].str.contains("furnished", case=False, na=False).astype(int)
        df["is_renovated"] = df["desc_nlp"].str.contains("renovated|newly", case=False, na=False).astype(int)

        # Text quality
        df["unique_word_count"] = df["desc_nlp"].apply(lambda x: len(set(str(x).split())))
        df["avg_word_len"] = df["desc_nlp"].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if isinstance(x, str) and len(x) > 0 else 0
        )

    # END
    return df

country = "usa"
df = pd.read_csv(f"{country}/{country}_cleaned_listings.csv")
unique_amenities = set()
engineered_df = add_features(df)

engineered_df.to_csv(f"{country}/{country}_engineered_listings.csv", index=False)
