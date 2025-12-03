import re
import os
import numpy as np
import pandas as pd
import ast


def create_polynomial_features(df, numeric_cols, degree=2):
    """
    Create polynomial features for key numeric variables
    """
    key_features = ["area", "bedrooms", "bathrooms", "property_age"]
    available_features = [col for col in key_features if col in numeric_cols]

    if len(available_features) >= 2:
        # Create interaction terms
        for i, col1 in enumerate(available_features):
            for col2 in available_features[i + 1 :]:
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]

    return df


def split_location(text, country):
    """
    Extract structured location info depending on country:
      - Iraq: "City, Governorate"
      - Lebanon: "City, Governorate, District"
      - USA: already split by state and city, no change.
    """

    parts = [p.strip() for p in str(text).split(",")]

    # -------------------------------------------
    # IRAQ (example: "Baghlumnara, Erbil")
    # -------------------------------------------
    if country == "USA":
        pass

    if country == "Iraq":
        if len(parts) >= 2:
            return pd.Series(
                {
                    "city": parts[1],
                }
            )
        else:
            return pd.Series({"city": parts[1]})

    # -------------------------------------------
    # LEBANON (example: "Zalka, Mount Lebanon Governorate, El Metn district")
    # -------------------------------------------
    if country == "Lebanon":
        if len(parts) == 3:
            return pd.Series(
                {
                    "city": parts[0],
                    "district": parts[2]
                    .replace("district", "")
                    .replace("districts", ""),
                }
            )
        elif len(parts) == 2:
            return pd.Series({"city": parts[0], "district": "NA"})
        else:
            return pd.Series({"city": parts[0], "district": "NA"})

    # -------------------------------------------
    # USA or anything else → no custom split
    # -------------------------------------------
    return pd.Series({"city": "NA", "district": "NA"})


def add_features(df):
    df = df.copy()

    # REMOVE DUPLICATE COLUMNS (keeps first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    if "location" in df.columns and "country" in df.columns:
        location_expanded = df.apply(
            lambda row: split_location(row["location"], row["country"]), axis=1
        )
        df = pd.concat([df, location_expanded], axis=1)

    # REMOVE DUPLICATE COLUMNS (keeps first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    if "description" in df.columns:
        df["description_word_count"] = (
            df["description"].astype(str).apply(lambda x: len(x.split()))
        )

    if "title" in df.columns:
        df["title_word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))

    if "year_built" in df.columns:
        df["property_age"] = 2025 - df["year_built"]

    country = df["country"][0]

    if country == "iraq":
        types = [
            "Agricultural Plot",
            "Apartment",
            "Commercial Building",
            "Commercial Floor",
            "Commercial House",
            "Commercial Plot",
            "Commercial Property",
            "Farm",
            "Hotel",
            "House",
            "Industrial Land",
            "Office",
            "Residential Building",
            "Residential Plot",
            "Restaurant",
            "Shop",
            "Warehouse",
        ]
        cities = [
            "Baghdad",
            "Duhok",
            "Erbil",
            "Halabja Shahid",
            "Kalar",
            "Kirkuk",
            "Najaf",
            "Shaqlawa",
            "Sulaymaniyah",
        ]
    elif country == "lebanon":
        types = [
            "Apartment",
            "Chalet",
            "Duplex",
            "Full floor",
            "Half floor",
            "Hotel Apartments",
            "Land",
            "Loft",
            "Penthouse",
            "Townhouse",
            "Triplex",
            "Villa",
            "Whole Building",
        ]
        cities = [
            "Aabrine",
            "Aamchit",
            "Aannaya",
            "Aaoukar",
            "Aatchane",
            "Abediyeh",
            "Abou Samra",
            "Achrafieh",
            "Adlieh",
            "Adma",
            "Adonis",
            "Ain Aalaq",
            "Ain Aar",
            "Ain Al Jdide",
            "Ain El Remmaneh",
            "Ain El Rihany",
            "Ain Najm",
            "Ain Saadeh",
            "Ain al-Mraiseh",
            "Aintoura",
            "Aitat",
            "Ajaltoun",
            "Al Mina",
            "Aley",
            "Antelias",
            "Aramoun",
            "Aylout",
            "Azmi",
            "Baabda",
            "Baabdat",
            "Badaro",
            "Bakich",
            "Ballouneh",
            "Baouchriyeh",
            "Barbara",
            "Basbina",
            "Batrakieh",
            "Batroun",
            "Bchamoun",
            "Bchelli",
            "Beit Chabeb",
            "Beit El Kikko",
            "Beit Meri",
            "Beit Misk",
            "Beit el Chaar",
            "Bhamdoun",
            "Bhamdoun Al Mhatta",
            "Bharsaf",
            "Biaqout",
            "Bikfaiya",
            "Bir Hassan",
            "Blat",
            "Bliss",
            "Bolonia",
            "Bouar",
            "Boulevard",
            "Bourj Hammoud",
            "Boutchay",
            "Bqaatouta",
            "Bqennaya",
            "Braij",
            "Brazilia",
            "Broummana",
            "Bsaba",
            "Bsalim",
            "Bteghrine",
            "Btouratij",
            "Burj Abi Haidar",
            "Bziza",
            "Chabtine",
            "Charehbil",
            "Charles Malek",
            "Chbaniyeh",
            "Chemlan",
            "Chiyah",
            "Choueifat",
            "Chouit",
            "Clemenceau",
            "Dahr El Souane",
            "Dam Wa Farz",
            "Damour",
            "Daroun Harissa",
            "Dawhet El Hoss",
            "Daychounieh",
            "Dbayeh",
            "Dbayeh Waterfront",
            "Dekwaneh",
            "Dhour Choueir",
            "Dik El Mehdi",
            "Douar",
            "Down Town",
            "El Biyada",
            "El Khall�",
            "Elissar",
            "Faitroun",
            "Faiyadiyeh",
            "Fanar",
            "Faqra",
            "Faraya",
            "Fassouh",
            "Fatqa",
            "Fghal",
            "Fidar",
            "Furn El Chebbak",
            "Gemayzeh",
            "Ghadir",
            "Ghazir",
            "Gherfine",
            "Hadad",
            "Hadath",
            "Hadirat",
            "Halat",
            "Hamra",
            "Haret El Sett",
            "Haret Sakher",
            "Hasbaiyya Baabda",
            "Hazmieh",
            "Hbaline",
            "Hboub",
            "Hbous",
            "Horch Tabet",
            "Hosrayel",
            "Jal el Dib",
            "Jamhour",
            "Jbeil",
            "Jdeideh",
            "Jeddayel",
            "Jeita",
            "Jisr El Bacha",
            "Jnah",
            "Jouret Al Ballout",
            "Kaakour",
            "Kafarakab",
            "Kaskas",
            "Kfar Aabida",
            "Kfardebian",
            "Kfarhbab",
            "Kfarshima",
            "Khenchara",
            "Klayaat",
            "Kniseh",
            "Koraytem",
            "Kornet Chehwan",
            "Koubba",
            "Koura",
            "Kousba",
            "Laqlouq",
            "Louaizeh",
            "Maaysrah",
            "Mahmerch",
            "Manara",
            "Mansourieh",
            "Mar Chaaya",
            "Mar Elias",
            "Mar Mikhael",
            "Mar Moussa El Douar",
            "Mar Roukoz",
            "Mar Takla",
            "Masqa",
            "Mastita",
            "Mazraat Yachouh",
            "Mechmech",
            "Mechref",
            "Mezher",
            "Monot",
            "Monteverde",
            "Mrouj",
            "Msaytbeh",
            "Mtayleb",
            "Naas",
            "Nabay",
            "Nahr Ibrahim",
            "Naqqache",
            "New Mar Takla",
            "Ouyoun Broummana",
            "Qalaa",
            "Qanat Bakish",
            "Qennabet Broummana",
            "Qornet El Hamra",
            "Qortadah",
            "Rabieh",
            "Rabweh",
            "Ramlet al-Baydah",
            "Ras Beirut",
            "Ras El Jabal",
            "Ras El Nabeh",
            "Rawabi",
            "Rawche",
            "Rihaniyeh",
            "Roumieh",
            "Rwayset Sawfar",
            "Sabtiyeh",
            "Sad El Baouchriyeh",
            "Safra",
            "Sahel Alma",
            "Saifi",
            "Salima",
            "Sanayeh",
            "Sarba",
            "Sawfar",
            "Sehayleh",
            "Sin El Fil",
            "Sioufi",
            "Sodeco",
            "Spears",
            "Tabaris",
            "Tabarja",
            "Tariq El Jdideh",
            "Tayyouneh",
            "Thoum",
            "Tilal Ain Saade",
            "Tripoli",
            "Verdun",
            "Wadi Chahrour",
            "Yarzeh",
            "Zaarour",
            "Zahle",
            "Zalka",
            "Zandouqah",
            "Zaraoun",
            "Zebdine",
            "Zikrit",
            "Zouk Mikael",
            "Zouk Mosbeh",
            "abed al-wahab inglizi",
            "ain al-tineh",
            "caracas",
            "lycee",
            "minet al-hoson",
            "sakiet al-janzeer",
            "tallet al-khayat",
        ]
    else:
        types = [
            "Condo",
            "Condo (co-op)",
            "Manufactured",
            "Moorage",
            "Multi-family",
            "Parking",
            "Single Family Residence, 24 - Floating Home/On-Water Res",
            "Single-family",
            "Single-family (co-op)",
            "Townhome",
            "Townhome (co-op)",
            "Vacant land",
        ]
        cities = [
            "Chicago",
            "Denver",
            "Kansas-City",
            "Las-Vegas",
            "Los-Angeles",
            "Miami",
            "Nashville",
            "New-York",
            "San-Diego",
            "Seattle",
            "Tampa",
            "Washington-DC",
        ]

    if "type" in df.columns:
        # Normalize the values
        df["type"] = df["type"].str.lower().str.strip()

        # Create dummies only for values that exist
        dummies = pd.get_dummies(df["type"], prefix="ptype", dtype=int)

        # Ensure ALL expected columns exist (fill missing with 0)
        for ptype in types:
            col_name = f"ptype_{ptype}"
            if col_name not in dummies.columns:
                dummies[col_name] = 0

        # Concatenate and drop original
        df = pd.concat([df, dummies], axis=1)
        # df = df.drop(columns=["type"])py

    if "city" in df.columns:
        # Normalize the values
        df["city"] = df["city"].str.lower().str.strip()

        # Create dummies only for values that exist
        dummies = pd.get_dummies(df["city"], prefix="pcity", dtype=int)

        # Ensure ALL expected columns exist (fill missing with 0)
        for pcity in cities:
            col_name = f"pcity_{pcity}"
            if col_name not in dummies.columns:
                dummies[col_name] = 0

        # Concatenate and drop original
        df = pd.concat([df, dummies], axis=1)
        # df = df.drop(columns=["city"])

    if "amenities" in df.columns:
        unique_amenities = set()

        def parse_amenities(cell):
            if isinstance(cell, list):
                return cell
            if pd.isna(cell):
                return []
            try:
                return ast.literal_eval(cell)
            except:
                return []

        # Create a temporary series of lists
        amenities_lists = df["amenities"].apply(parse_amenities)

        df["num_amenities"] = amenities_lists.apply(len)

        for am_list in amenities_lists:
            unique_amenities.update(am_list)

        for group in unique_amenities:
            df[f"amenity_{group}"] = amenities_lists.apply(lambda x: int(group in x))

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

    # ---- Guarantee desc_nlp exists ----
    if "desc_nlp" not in df.columns:
        if "description" in df.columns:
            df["desc_nlp"] = df["description"].fillna("").astype(str).str.lower()
        else:
            df["desc_nlp"] = ""

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
        # df["has_luxury"] = df["amenities_old"].apply(
        #     lambda x: int(any(a in x for a in lux)) if isinstance(x, (str, list)) else 0
        # )
        pass

    elif country == "USA":
        df["is_new_property"] = (
            df["property_age"] < df["property_age"].median()
        ).astype(int)
    else:
        pass

    # ----------------------------------------
    # 6.1 STRUCTURAL RATIO FEATURES
    # ----------------------------------------
    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
        df["area_per_room"] = df["area"] / (df["total_rooms"].replace(0, np.nan))
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)

    if "bedrooms" in df.columns:
        df["is_family_sized"] = (df["bedrooms"] >= 3).astype(int)
        df["is_studio_flag"] = (
            (df["bedrooms"] <= 1) & (df["area"] < df["area"].median())
        ).astype(int)

    # ----------------------------------------
    # 6.2 POLYNOMIAL & INTERACTION FEATURES (NEW)
    # ----------------------------------------
    if "area" in df.columns:
        df["area_squared"] = df["area"] ** 2
        df["log_area_feat"] = np.log1p(df["area"])

    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bed_bath_interaction"] = df["bedrooms"] * df["bathrooms"]
        df["room_density"] = (df["bedrooms"] + df["bathrooms"]) / df["area"].replace(
            0, 1
        )

    # ----------------------------------------
    # 6.3 NLP-BASED QUALITY SIGNALS
    # ----------------------------------------
    if "desc_nlp" in df.columns:
        df["has_view"] = (
            df["desc_nlp"].str.contains("view", case=False, na=False).astype(int)
        )
        df["is_furnished"] = (
            df["desc_nlp"].str.contains("furnished", case=False, na=False).astype(int)
        )
        df["is_renovated"] = (
            df["desc_nlp"]
            .str.contains("renovated|newly", case=False, na=False)
            .astype(int)
        )

        # Text quality
        df["unique_word_count"] = df["desc_nlp"].apply(
            lambda x: len(set(str(x).split()))
        )
        df["avg_word_len"] = df["desc_nlp"].apply(
            lambda x: (
                np.mean([len(w) for w in str(x).split()])
                if isinstance(x, str) and len(x) > 0
                else 0
            )
        )

        # Keep only numeric columns and handle remaining missing values
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Create polynomial features
    numeric_df = create_polynomial_features(numeric_df, numeric_df.columns)

    # END
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = [
        {
            "input": "iraq/iraq_improved_cleaned_listings.csv",
            "output": "iraq/iraq_engineered_listings.csv",
        },
        {
            "input": "lebanon/lebanon_improved_cleaned_listings.csv",
            "output": "lebanon/lebanon_engineered_listings.csv",
        },
        {
            "input": "usa/usa_improved_cleaned_listings.csv",
            "output": "usa/usa_engineered_listings.csv",
        },
    ]

    for ds in datasets:
        input_path = os.path.join(base_dir, ds["input"])
        output_path = os.path.join(base_dir, ds["output"])

        if os.path.exists(input_path):
            print(f"Processing {ds['input']}...")
            df = pd.read_csv(input_path)

            # Ensure country column is present and consistent for the logic
            if "country" not in df.columns or df["country"].isnull().all():
                # Fallback if country is missing, though cleaning should have fixed it
                if "iraq" in ds["input"].lower():
                    df["country"] = "IRAQ"
                elif "lebanon" in ds["input"].lower():
                    df["country"] = "LEBANON"
                elif "usa" in ds["input"].lower():
                    df["country"] = "USA"

            engineered_df = add_features(df)
            engineered_df.to_csv(output_path, index=False)
            print(f"Saved engineered data to {ds['output']}")
        else:
            print(f"File not found: {ds['input']}")
