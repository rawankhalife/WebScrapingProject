import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import your cleaning and feature engineering functions
from improved_cleaning import improved_data_cleaning, predicting_cleaning
from feature_engineering import add_features

area_limits = {
    "IRAQ": (50, 10000),
    "LEBANON": (50, 15000),
    "USA": (50, 50000),
}

# Country-specific options
country_cities = {
    "IRAQ": [
        "Baghdad",
        "Duhok",
        "Erbil",
        "Halabja Shahid",
        "Kalar",
        "Kirkuk",
        "Najaf",
        "Shaqlawa",
        "Sulaymaniyah",
    ],
    "LEBANON": [
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
    ],
    "USA": [
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
    ],
}

country_property_types = {
    "IRAQ": [
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
    ],
    "LEBANON": [
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
    ],
    "USA": [
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
    ],
}

country_amenities = {
    "IRAQ": [
        "Swimming Pool",
        "Parking Spaces",
        "Central Air-Conditioner",
        "Gym",
        "Elevator",
        "Balcony",
        "Security Staff",
        "Pets Allowed",
        "Green Area",
        "Internet",
        "Sauna",
        "Satellite/Cable TV",
        "Maintenance",
        "Furnished Kitchen",
        "Double Glazed Windows",
        "School",
        "Mosque",
        "Market",
        "Restaurant",
        "Basement",
    ],
    "LEBANON": [
        "Balcony",
        "Terrace",
        "Shared Pool",
        "Shared Gym",
        "Private Pool",
        "Private Garden",
        "Covered Parking",
        "Built in Wardrobes",
        "Central A/C",
        "Kitchen Appliances",
        "Maids Room",
        "Security",
        "View of Water",
        "View of Landmark",
        "Walk-in Closet",
        "Storage Room",
        "Concierge",
        "Lobby in Building",
        "Near restaurants",
        "Near Public transportation",
    ],
    "USA": [
        "Pool",
        "Parking",
        "Electric",
        "Gas",
        "Heating",
        "Cooling",
        "Fireplace",
        "Internet",
        "Sewer",
        "Water",
        "Air Conditioning",
        "Cable Connected",
        "Interior Features",
        "Exterior Features",
        "Utilities",
        "Water Heater Type",
        "Water Source",
        "Sewer Description",
        "Power Company",
        "Municipal Trash",
    ],
}

st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 0;}
    .sub-header {text-align: center; color: #666; font-size: 1.2rem; margin-top: -10px; margin-bottom: 30px;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;}
    .stButton>button {width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px; font-size: 18px; font-weight: bold; border-radius: 10px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_scalers():
    models = {}
    scalers = {}
    feature_lists = {}
    countries = ["iraq", "lebanon", "usa"]
    for country in countries:
        try:
            models[country] = joblib.load(f"{country}/{country}_improved_model.pkl")
            scalers[country] = joblib.load(f"{country}/{country}_scaler.pkl")
            feature_lists[country] = joblib.load(
                f"{country}/{country}_improved_features.pkl"
            )
        except FileNotFoundError:
            try:
                models[country] = joblib.load(f"{country}/{country}_xgb_model.pkl")
                feature_lists[country] = joblib.load(
                    f"{country}/{country}_feature_list.pkl"
                )
                scalers[country] = None
            except:
                pass
    return models, scalers, feature_lists


@st.cache_data
def load_property_data():
    datasets = {}
    countries = ["iraq", "lebanon", "usa"]
    for country in countries:
        try:
            datasets[country] = pd.read_csv(
                f"{country}/{country}_engineered_listings.csv"
            )
        except FileNotFoundError:
            try:
                datasets[country] = pd.read_csv(
                    f"{country}/{country}_improved_cleaned_listings.csv"
                )
            except FileNotFoundError:
                datasets[country] = pd.DataFrame()
    return datasets


def calculate_market_metrics(datasets):
    metrics = {}
    for country, df in datasets.items():
        if df.empty:
            continue
        country_upper = country.upper()
        metrics[country_upper] = {
            "total_properties": len(df),
            "avg_price": df["price"].mean() if "price" in df.columns else 0,
            "median_price": df["price"].median() if "price" in df.columns else 0,
            "min_price": df["price"].min() if "price" in df.columns else 0,
            "max_price": df["price"].max() if "price" in df.columns else 0,
            "std_price": df["price"].std() if "price" in df.columns else 0,
            "avg_area": df["area"].mean() if "area" in df.columns else 0,
            "price_per_sqft": (
                (df["price"].mean() / df["area"].mean())
                if "area" in df.columns and df["area"].mean() > 0
                else 0
            ),
            "avg_bedrooms": df["bedrooms"].mean() if "bedrooms" in df.columns else 0,
            "avg_bathrooms": df["bathrooms"].mean() if "bathrooms" in df.columns else 0,
            "dataframe": df,
        }
    return metrics


def get_country_specific_analytics(df, country):
    analytics = {}
    if df.empty:
        return analytics

    if "type" in df.columns or "property_type" in df.columns:
        type_col = "type" if "type" in df.columns else "property_type"
        analytics["price_by_type"] = (
            df.groupby(type_col)["price"].agg(["mean", "count", "median"]).reset_index()
        )

    if "city" in df.columns:
        analytics["price_by_city"] = (
            df.groupby("city")["price"].agg(["mean", "count", "median"]).reset_index()
        )
        analytics["price_by_city"] = (
            analytics["price_by_city"].sort_values("mean", ascending=False).head(10)
        )

    if "bedrooms" in df.columns:
        analytics["bedroom_distribution"] = df["bedrooms"].value_counts().reset_index()
        analytics["bedroom_distribution"].columns = ["bedrooms", "count"]

    if "area" in df.columns and "price" in df.columns:
        analytics["area_price_corr"] = df[["area", "price"]].corr().iloc[0, 1]

    if "price" in df.columns:
        analytics["price_quartiles"] = {
            "Q1": df["price"].quantile(0.25),
            "Q2": df["price"].quantile(0.50),
            "Q3": df["price"].quantile(0.75),
        }

    return analytics


def search_properties(query, country_data, top_n=5):
    if country_data.empty or not query:
        return pd.DataFrame()
    query_lower = query.lower()
    search_columns = [
        "description",
        "title",
        "city",
        "type",
        "property_type",
        "location",
    ]
    available_columns = [col for col in search_columns if col in country_data.columns]
    if not available_columns:
        return pd.DataFrame()
    mask = pd.Series([False] * len(country_data))
    for col in available_columns:
        mask |= (
            country_data[col]
            .astype(str)
            .str.lower()
            .str.contains(query_lower, na=False)
        )
    results = country_data[mask].head(top_n)
    return results


# Load models and data
models, scalers, feature_lists = load_models_and_scalers()
property_datasets = load_property_data()
market_metrics = calculate_market_metrics(property_datasets)

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/real-estate.png", width=150)
    st.markdown("## 🏠 Real Estate AI")
    st.markdown("---")

    page = st.radio(
        "Navigation", ["🎯 Price Predictor", "📊 Market Analytics", "ℹ️ About"]
    )
    st.markdown("---")

    selected_country = st.selectbox(
        "Select Country", ["IRAQ", "LEBANON", "USA"], key="selected_country"
    )

    st.markdown("### 🔍 NLP Search")
    search_query = st.text_input(
        "Search properties:", placeholder="e.g., luxury apartment with pool"
    )

    if search_query:
        st.info(f"🔎 Searching in {selected_country}...")
        country_data = property_datasets.get(selected_country.lower(), pd.DataFrame())
        search_results = search_properties(search_query, country_data)

        if not search_results.empty:
            st.success(f"Found {len(search_results)} properties")
            for idx, row in search_results.iterrows():
                with st.expander(
                    f"📍 {row.get('city', 'N/A')} - ${row.get('price', 0):,.0f}"
                ):
                    prop_type = row.get("type", row.get("property_type", "N/A"))
                    st.write(f"**Type:** {prop_type}")
                    st.write(f"**Area:** {row.get('area', 'N/A')} sq ft")
                    st.write(f"**Bedrooms:** {row.get('bedrooms', 'N/A')}")
                    st.write(f"**Bathrooms:** {row.get('bathrooms', 'N/A')}")
                    if "description" in row and pd.notna(row["description"]):
                        st.write(f"**Description:** {str(row['description'])[:100]}...")
        else:
            st.warning("No properties found matching your search.")

    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    st.metric("🇮🇶 Iraq", "82.82%", "R² Score")
    st.metric("🇱🇧 Lebanon", "74.95%", "R² Score")
    st.metric("🇺🇸 USA", "58.23%", "R² Score")

    st.markdown("---")

    if selected_country in market_metrics:
        st.markdown(f"### 📊 {selected_country} Market")
        metrics = market_metrics[selected_country]
        st.metric("Properties", f"{metrics['total_properties']:,}")
        st.metric("Avg Price", f"${metrics['avg_price']:,.0f}")
        st.metric("Price/SqFt", f"${metrics['price_per_sqft']:.0f}")
        st.markdown("---")

    st.markdown("### 📈 Model Performance")
    st.metric("🇮🇶 Iraq", "82.82%", "R² Score")
    st.metric("🇱🇧 Lebanon", "74.95%", "R² Score")
    st.metric("🇺🇸 USA", "58.23%", "R² Score")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info(
        "✓ Fill all property details accurately\n\n✓ Select relevant amenities\n\n✓ Provide detailed description"
    )

# MAIN CONTENT
if page == "🎯 Price Predictor":
    st.markdown(
        '<p class="main-header">🏠 Real Estate Price Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Property Valuation with Advanced Machine Learning</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("### 📝 Property Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            country = st.selectbox(
                "🌍 Country",
                ["IRAQ", "LEBANON", "USA"],
                help="Select property location",
            )
            city = st.selectbox("🏙️ City", country_cities[country])
            if country == "USA":
                year_built = st.number_input(
                    "📅 Year Built", min_value=1800, max_value=2025, value=2010
                )

        with col2:
            bedrooms = st.number_input(
                "🛏️ Bedrooms", min_value=0, max_value=20, value=3, step=1
            )
            bathrooms = st.number_input(
                "🚿 Bathrooms", min_value=1, max_value=20, value=2, step=1
            )

        with col3:
            property_type = st.selectbox(
                "🏢 Property Type", country_property_types[country]
            )
            area = st.number_input(
                "📏 Area (sq ft)",
                min_value=area_limits[country][0],
                max_value=area_limits[country][1],
                value=1500,
                step=50,
            )

        if country != "USA":
            year_built = 2010

    st.markdown("### 📄 Property Description")
    description = st.text_area(
        "Description",
        value="Modern property with excellent amenities",
        height=120,
        help="Provide detailed property description",
    )

    st.markdown("### ✨ Select Amenities")
    selected_amenities = []
    num_cols = 4
    amenity_cols = st.columns(num_cols)

    for idx, amenity in enumerate(country_amenities[country]):
        col_idx = idx % num_cols
        with amenity_cols[col_idx]:
            if st.checkbox(amenity, key=f"amenity_{amenity}"):
                selected_amenities.append(amenity)

    st.markdown("---")

    if st.button("🔮 PREDICT PRICE", type="primary"):
        with st.spinner("Analyzing property data..."):
            if country.lower() in models:
                try:
                    input_data = pd.DataFrame(
                        [
                            {
                                "country": country,
                                "area": area,
                                "bedrooms": bedrooms,
                                "bathrooms": bathrooms,
                                "year_built": year_built,
                                "type": property_type,
                                "city": city,
                                "location": city,
                                "description": description,
                                "amenities": str(selected_amenities),
                            }
                        ]
                    )

                    cleaned_data = predicting_cleaning(input_data, country.lower())
                    processed_data = add_features(cleaned_data)

                    text_cols = [
                        "url",
                        "description",
                        "title",
                        "amenities",
                        "location",
                        "desc_nlp",
                        "title_nlp",
                        "city",
                        "country",
                        "amenities_old",
                        "district",
                        "state",
                        "property_type",
                        "type",
                        "address",
                    ]
                    for col in text_cols:
                        if col in processed_data.columns:
                            processed_data = processed_data.drop(columns=[col])

                    processed_data = processed_data.select_dtypes(include=[np.number])

                    if "price" in processed_data.columns:
                        processed_data = processed_data.drop(columns=["price"])
                    if "log_price" in processed_data.columns:
                        processed_data = processed_data.drop(columns=["log_price"])

                    feature_list = feature_lists[country.lower()]
                    aligned_data = processed_data.reindex(
                        columns=feature_list, fill_value=0
                    )

                    if scalers[country.lower()] is not None:
                        aligned_data_scaled = scalers[country.lower()].transform(
                            aligned_data
                        )
                        aligned_data = pd.DataFrame(
                            aligned_data_scaled, columns=aligned_data.columns
                        )

                    model = models[country.lower()]
                    log_prediction = model.predict(aligned_data)[0]

                    if log_prediction < 20:
                        prediction = np.expm1(log_prediction)
                    else:
                        prediction = log_prediction

                    st.success("✅ Prediction Complete!")

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(
                            f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                            <h2 style='margin: 0;'>Predicted Price</h2>
                            <h1 style='font-size: 3rem; margin: 10px 0;'>${prediction:,.0f}</h1>
                            <p style='margin: 0; opacity: 0.9;'>Estimated Market Value</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown("### 📊 Property Analytics")

                    col1, col2, col3, col4 = st.columns(4)
                    price_per_sqft = prediction / area

                    with col1:
                        st.metric("Price/sq ft", f"${price_per_sqft:.2f}")

                    with col2:
                        if country in market_metrics:
                            avg_market_price_per_sqft = market_metrics[country][
                                "price_per_sqft"
                            ]
                        else:
                            avg_market_price_per_sqft = price_per_sqft

                        diff = (
                            (
                                (price_per_sqft - avg_market_price_per_sqft)
                                / avg_market_price_per_sqft
                            )
                            * 100
                            if avg_market_price_per_sqft > 0
                            else 0
                        )
                        st.metric(
                            "vs Market Avg", f"{diff:+.1f}%", delta=f"{diff:.1f}%"
                        )

                    with col3:
                        if prediction < 100000:
                            category = "Affordable"
                        elif prediction < 500000:
                            category = "Mid-Range"
                        elif prediction < 1000000:
                            category = "Premium"
                        else:
                            category = "Luxury"
                        st.metric("Category", category)

                    with col4:
                        amenity_score = len(selected_amenities) * 5
                        st.metric("Amenity Score", f"{min(amenity_score, 100)}/100")

                    st.markdown("### 💰 Price Breakdown")
                    breakdown_data = {
                        "Component": [
                            "Base Price",
                            "Area Premium",
                            "Amenities",
                            "Location",
                        ],
                        "Value": [
                            prediction * 0.5,
                            prediction * 0.25,
                            prediction * 0.15,
                            prediction * 0.1,
                        ],
                    }

                    fig = px.pie(
                        breakdown_data,
                        values="Value",
                        names="Component",
                        color_discrete_sequence=px.colors.sequential.RdBu,
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    if st.checkbox("Show debug info"):
                        st.write(str(e))
            else:
                st.error(f"Model for {country} not available.")

elif page == "📊 Market Analytics":
    st.markdown(
        '<p class="main-header">📊 Market Analytics</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Real Estate Market Insights & Trends</p>',
        unsafe_allow_html=True,
    )

    # Create tabs for All Countries vs Selected Country
    tab1, tab2 = st.tabs(
        ["🌍 All Countries Overview", f"📍 {selected_country} Deep Dive"]
    )

    with tab1:
        # Calculate real aggregate metrics
        total_props = sum([m["total_properties"] for m in market_metrics.values()])
        avg_price_all = np.mean([m["avg_price"] for m in market_metrics.values()])
        avg_price_per_sqft = np.mean(
            [
                m["price_per_sqft"]
                for m in market_metrics.values()
                if m["price_per_sqft"] > 0
            ]
        )

        st.markdown("### 📈 Global Market Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Properties", f"{total_props:,}")
        with col2:
            st.metric("Avg Market Value", f"${avg_price_all:,.0f}")
        with col3:
            st.metric("Avg Price/SqFt", f"${avg_price_per_sqft:.0f}")

        st.markdown("---")

        # Real data from datasets
        countries_data = pd.DataFrame(
            {
                "Country": [k for k in market_metrics.keys()],
                "Avg Price": [m["avg_price"] for m in market_metrics.values()],
                "Properties": [m["total_properties"] for m in market_metrics.values()],
                "Avg Area": [m["avg_area"] for m in market_metrics.values()],
                "Price per SqFt": [
                    m["price_per_sqft"] for m in market_metrics.values()
                ],
            }
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 Average Property Prices")
            fig = px.bar(
                countries_data,
                x="Country",
                y="Avg Price",
                color="Country",
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
                text="Avg Price",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🏘️ Property Distribution")
            fig = px.pie(
                countries_data,
                values="Properties",
                names="Country",
                hole=0.4,
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Price per Square Foot")
            fig = px.bar(
                countries_data,
                x="Country",
                y="Price per SqFt",
                color="Price per SqFt",
                color_continuous_scale="Viridis",
                text="Price per SqFt",
            )
            fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📐 Area vs Price Correlation")
            fig = px.scatter(
                countries_data,
                x="Avg Area",
                y="Avg Price",
                size="Properties",
                color="Country",
                size_max=60,
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
            )
            fig.update_layout(height=400)
            fig.update_xaxes(title="Average Area (sq ft)")
            fig.update_yaxes(title="Average Price ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Detailed Market Statistics")
        display_data = countries_data.copy()
        display_data["Avg Price"] = display_data["Avg Price"].apply(
            lambda x: f"${x:,.0f}"
        )
        display_data["Avg Area"] = display_data["Avg Area"].apply(
            lambda x: f"{x:,.0f} sq ft"
        )
        display_data["Price per SqFt"] = display_data["Price per SqFt"].apply(
            lambda x: f"${x:.0f}"
        )
        st.dataframe(display_data, use_container_width=True, hide_index=True)

    with tab2:
        # Country-specific deep dive
        if selected_country in market_metrics:
            df = market_metrics[selected_country]["dataframe"]
            analytics = get_country_specific_analytics(df, selected_country)

            # Filters
            st.markdown("### 🔧 Filter Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                price_range = st.slider(
                    "Price Range ($)",
                    float(df["price"].min()),
                    float(df["price"].max()),
                    (float(df["price"].min()), float(df["price"].max())),
                )

            with col2:
                if "bedrooms" in df.columns:
                    bedroom_filter = st.multiselect(
                        "Bedrooms", sorted(df["bedrooms"].unique()), default=None
                    )

            with col3:
                if "city" in df.columns:
                    city_filter = st.multiselect(
                        "Cities", sorted(df["city"].unique()), default=None
                    )

            # Apply filters
            filtered_df = df[
                (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])
            ]
            if bedroom_filter:
                filtered_df = filtered_df[filtered_df["bedrooms"].isin(bedroom_filter)]
            if city_filter:
                filtered_df = filtered_df[filtered_df["city"].isin(city_filter)]

            st.info(f"Showing {len(filtered_df):,} of {len(df):,} properties")

            st.markdown("---")

            # Key metrics for selected country
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Price", f"${filtered_df['price'].mean():,.0f}")
            with col2:
                st.metric("Median Price", f"${filtered_df['price'].median():,.0f}")
            with col3:
                st.metric("Avg Area", f"{filtered_df['area'].mean():,.0f} sq ft")
            with col4:
                if "bedrooms" in filtered_df.columns:
                    st.metric("Avg Bedrooms", f"{filtered_df['bedrooms'].mean():.1f}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            # Price by Property Type
            if "price_by_type" in analytics:
                with col1:
                    st.markdown("### 🏘️ Price by Property Type")
                    type_data = (
                        analytics["price_by_type"]
                        .sort_values("mean", ascending=False)
                        .head(10)
                    )
                    fig = px.bar(
                        type_data,
                        x=type_data.columns[0],
                        y="mean",
                        color="mean",
                        color_continuous_scale="Blues",
                        text="mean",
                        labels={"mean": "Avg Price"},
                    )
                    fig.update_traces(
                        texttemplate="$%{text:,.0f}", textposition="outside"
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            # Price by City
            if "price_by_city" in analytics:
                with col2:
                    st.markdown("### 🌆 Top 10 Cities by Price")
                    fig = px.bar(
                        analytics["price_by_city"],
                        x="city",
                        y="mean",
                        color="mean",
                        color_continuous_scale="Reds",
                        text="mean",
                        labels={"mean": "Avg Price"},
                    )
                    fig.update_traces(
                        texttemplate="$%{text:,.0f}", textposition="outside"
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Bedroom distribution
            if "bedroom_distribution" in analytics:
                st.markdown("### 🛏️ Bedroom Distribution")
                fig = px.bar(
                    analytics["bedroom_distribution"],
                    x="bedrooms",
                    y="count",
                    color="count",
                    color_continuous_scale="Viridis",
                    text="count",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Area vs Price scatter
            if "area" in filtered_df.columns and "price" in filtered_df.columns:
                st.markdown("### 📐 Area vs. Price")
                fig = px.scatter(
                    filtered_df,
                    x="area",
                    y="price",
                    trendline="ols",
                    opacity=0.7,
                    color_discrete_sequence=["#764ba2"],
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Correlation heatmap
            st.markdown("### 🔥 Correlation Heatmap")
            numeric_cols = filtered_df.select_dtypes(include=["number"])
            if len(numeric_cols.columns) > 1:
                corr = numeric_cols.corr()
                fig = px.imshow(
                    corr, text_auto=True, color_continuous_scale="RdBu_r", height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numerical data to compute correlations.")

            st.markdown("---")

            # Preview filtered dataset
            st.markdown("### 📋 Filtered Data Preview")
            st.dataframe(filtered_df.head(50), use_container_width=True)

        else:
            st.error("No data available for the selected country.")
else:  # About page
    st.markdown('<p class="main-header">ℹ️ About This App</p>', unsafe_allow_html=True)

    st.markdown(
        """
    ## 🚀 Advanced Real Estate Price Prediction
    
    This application leverages cutting-edge machine learning techniques to provide accurate property valuations.
    
    ### ✨ Key Features
    - **HIGH Accuracy** across all supported markets
    - **Advanced NLP** for description analysis
    - **Real-time predictions** with instant results
    - **Multi-country support** (Iraq, Lebanon, USA)
    
    ### 🔧 Technology Stack
    - **XGBoost** for gradient boosting
    - **Polynomial Features** for complex interactions
    - **RobustScaler** for outlier-resistant normalization
    - **Cross-validation** for reliable performance
    
    ### 📊 Model Performance
    """
    )

    performance_data = pd.DataFrame(
        {
            "Country": ["Iraq", "Lebanon", "USA"],
            "R² Score": [0.8282, 0.7495, 0.5823],
            "RMSE": [9952, 82220, 94193],
        }
    )

    st.dataframe(performance_data, use_container_width=True)
