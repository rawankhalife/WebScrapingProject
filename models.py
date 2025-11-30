import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


COUNTRY = "lebanon"
df = pd.read_csv(f"{COUNTRY}/{COUNTRY}_new_engineered_listings.csv")   

# ======================================================
# we tried removing properties costing more than a million the performance got better
# ======================================================
# df = df[df["price"] <= 1_000_000].copy()


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

# Save feature list for Streamlit
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
# 5. TRAIN XGBOOST
# ======================================================
model = XGBRegressor(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    early_stopping_rounds=50,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ====== METRICS AFTER TRAINING ======
results = model.evals_result()

print("\n===== TRAINING LOG (first 20 rounds) =====")
for i in range(min(20, len(results['validation_0']['rmse']))):
    print(f"Round {i:4d}  |  Val RMSE: {results['validation_0']['rmse'][i]:.5f}")

print("\n===== BEST MODEL INFO =====")
print("Best iteration:", model.best_iteration)
print("Best RMSE:", model.best_score)

print("\n===== LEARNING RATE =====")
print("Learning rate:", model.learning_rate)

print("\n===== TREE INFO =====")
print("Trees used:", model.best_iteration + 1)
print("Trees allowed:", model.n_estimators)

print("\n===== MODEL PARAMETERS =====")
print(model.get_params())

# Save model
joblib.dump(model, f"{save_dir}/{country.lower()}_xgb_model.pkl")
print(f"Saved model as {country.lower()}_xgb_model.pkl")

# ======================================================
# 6. EVALUATE
# ======================================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ======================================================
# SAVE FULL METRICS + TRAINING LOG + MODEL PARAMS
# ======================================================

results = model.evals_result()

# --- FIRST 20 VALIDATION ROUNDS ---
training_log = "===== TRAINING LOG (first 20 rounds) =====\n"
for i in range(min(20, len(results['validation_0']['rmse']))):
    training_log += f"Round {i:4d}  |  Val RMSE: {results['validation_0']['rmse'][i]:.5f}\n"

# --- BEST MODEL INFO ---
best_info = f"""
===== BEST MODEL INFO =====
Best iteration: {model.best_iteration}
Best RMSE (Validation): {model.best_score}
"""

# --- LEARNING RATE ---
learn_info = f"""
===== LEARNING RATE =====
Learning rate: {model.learning_rate}
"""

# --- TREE INFO ---
tree_info = f"""
===== TREE INFO =====
Trees used (best iteration): {model.best_iteration + 1}
Trees allowed (n_estimators): {model.n_estimators}
"""

# --- MODEL PARAMETERS ---
params_info = "===== MODEL PARAMETERS =====\n"
for k, v in model.get_params().items():
    params_info += f"{k}: {v}\n"

# --- FINAL PERFORMANCE ---
perf_info = f"""
===== MODEL PERFORMANCE ({country}) =====
RMSE: {rmse:,.2f}
MAE : {mae:,.2f}
R²  : {r2:.4f}
"""

# ---- COMBINE ALL SECTIONS ----
full_report = (
    training_log
    + "\n"
    + best_info
    + learn_info
    + tree_info
    + params_info
    + perf_info
)

# ---- SAVE FILE ----
filename = f"{country.lower()}_full_metrics.txt"

with open(f"{save_dir}/{filename}", "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\nSaved FULL training report to {filename}")

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE : {mae:,.2f}")
print(f"R²  : {r2:.4f}")

# ======================================================
# 7. FEATURE IMPORTANCE PLOT
# ======================================================
plt.figure(figsize=(10, 8))
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

plt.barh(X.columns[indices][:20], importance[indices][:20])
plt.gca().invert_yaxis()
plt.title(f"Top 20 Feature Importances ({country})")
plt.savefig(f"{save_dir}/feature_importances.png", dpi=200, bbox_inches="tight")
plt.show()