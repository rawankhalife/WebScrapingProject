import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor  


# ======================================================
# 1. LOAD CLEANED + ENGINEERED DATA
# ======================================================
COUNTRY = "iraq"
df = pd.read_csv(f"{COUNTRY}/{COUNTRY}_engineered_listings.csv")   

# ======================================================
# DETECT COUNTRY
# ======================================================
if "country" not in df.columns:
    raise ValueError("Dataset must contain a 'country' column.")

country = df["country"].iloc[0].upper()
print(f"\nDetected Country: {country}")

save_dir = country.capitalize()  

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"All output files will be saved to: {save_dir}/")

# ======================================================
# 2. COUNTRY-SPECIFIC DROP COLS
# ======================================================

if country == "IRAQ":
    drop_cols = [
        "url", "description", "title", "amenities", "location",
        "desc_nlp", "title_nlp",
        "city", "country",
        "amenities_old" if "amenities_old" in df.columns else None
    ]

elif country == "LEBANON":
    drop_cols = [
        "url", "description", "title", "amenities", "location",
        "desc_nlp", "title_nlp",
        "city", "district", "country",
        "amenities_old" if "amenities_old" in df.columns else None
    ]

elif country == "USA":
    drop_cols = [
        "url", "description", "title", "amenities", "location",
        "desc_nlp", "title_nlp",
        "city", "state", "country",
    ]

else:
    raise ValueError("Unknown country encountered!")

drop_cols = [c for c in drop_cols if c]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ======================================================
# 3. ENSURE NUMERIC ONLY
# ======================================================
df = df.select_dtypes(include=[np.number]).fillna(0)

target = "price"
X = df.drop(columns=[target])
y = df[target]

feature_list = list(X.columns)
joblib.dump(feature_list, f"{save_dir}/{country.lower()}_feature_list.pkl")
print("Saved feature list:", feature_list)

# ======================================================
# 4. TRAIN / VAL / TEST SPLIT (70 / 10 / 20)
# ======================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, random_state=42
)

# ======================================================
# 5. TRAIN RANDOM FOREST
# ======================================================
model = RandomForestRegressor(
    n_estimators=600,         
    max_depth=None,     
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,             
)

model.fit(X_train, y_train)

print("\n===== MODEL INFO (Random Forest) =====")
print("Trees:", model.n_estimators)
print("Max Depth:", model.max_depth)
print("Features:", len(feature_list))

# Save model
joblib.dump(model, f"{save_dir}/{country.lower()}_rf_model.pkl")
print(f"Saved model as {country.lower()}_rf_model.pkl")

# ======================================================
# 6. EVALUATE
# ======================================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ======================================================
# SAVE FULL METRICS REPORT
# ======================================================
report = f"""
===== RANDOM FOREST MODEL PERFORMANCE ({country}) =====
RMSE: {rmse:,.2f}
MAE : {mae:,.2f}
R²  : {r2:.4f}

===== MODEL PARAMETERS =====
n_estimators: {model.n_estimators}
max_depth:    {model.max_depth}
max_features: {model.max_features}
min_samples_split: {model.min_samples_split}
min_samples_leaf:  {model.min_samples_leaf}
bootstrap: {model.bootstrap}
"""

filename = f"{country.lower()}_rf_full_metrics.txt"
with open(f"{save_dir}/{filename}", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"Saved FULL training report to {filename}")

# ======================================================
# 7. FEATURE IMPORTANCE PLOT
# ======================================================
plt.figure(figsize=(10, 8))
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.barh(X.columns[indices][:20], importance[indices][:20])
plt.gca().invert_yaxis()
plt.title(f"Top 20 Feature Importances (RF - {country})")
plt.savefig(f"{save_dir}/rf_feature_importances.png", dpi=200, bbox_inches="tight")
plt.show()
