import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


def train_improved_model(country_name):
    """
    Train improved model with better data processing and validation
    """
    print(f"\n\n{'='*60}")
    print(f"IMPROVED TRAINING FOR: {country_name.upper()}")
    print(f"{'='*60}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        base_dir, country_name, f"{country_name}_engineered_listings.csv"
    )

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # ======================================================
    # 1. IMPROVED DATA CLEANING AND FEATURE ENGINEERING
    # ======================================================
    # Country-specific column drops
    country = country_name.upper()
    if country == "IRAQ":
        drop_cols = [
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
            "price_per_sqft",
            "type",
        ]
    elif country == "LEBANON":
        drop_cols = [
            "url",
            "description",
            "title",
            "amenities",
            "location",
            "desc_nlp",
            "title_nlp",
            "city",
            "district",
            "country",
            "price_per_sqft",
            "type",
        ]
    elif country == "USA":
        drop_cols = [
            "url",
            "description",
            "title",
            "amenities",
            "address",
            "parking",
            "desc_nlp",
            "title_nlp",
            "city",
            "state",
            "country",
            "price_per_sqft",
            "property_type",
        ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.select_dtypes(include=[np.number])

    # ======================================================
    # 2. PREPARE TARGET AND FEATURES
    # ======================================================
    target = "price"
    if target not in df.columns:
        print(f"Target column '{target}' not found!")
        return

    # Log transform the target for better distribution
    y_raw = df[target].values
    y = np.log1p(y_raw)  # log(1+x) transformation

    # Prepare features
    X = df.drop(columns=[target])

    # ======================================================
    # 3. ROBUST SCALING
    # ======================================================
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # ======================================================
    # 4. TRAIN/TEST SPLIT
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=None
    )

    # ======================================================
    # 5. MODEL TRAINING WITH PROPER HYPERPARAMETER TUNING
    # ======================================================

    # XGBoost parameters
    xgb_param_dist = {
        "n_estimators": [500, 1000, 1500, 2000],
        "max_depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
    }

    # Random Forest parameters
    rf_param_dist = {
        "n_estimators": [300, 500, 800, 1000],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8, 0.9],
    }

    models = {
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
            xgb_param_dist,
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_dist,
        ),
    }

    best_model = None
    best_score = float("inf")
    best_model_name = ""

    results = {}

    for model_name, (model, param_dist) in models.items():
        print(f"\nTraining {model_name}...")

        # RandomizedSearchCV with more iterations
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=30,  # More iterations for better search
            cv=5,  # 5-fold cross-validation
            scoring="neg_root_mean_squared_error",
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_log = search.best_estimator_.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # Inverse log transform
        y_test_actual = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)

        results[model_name] = {
            "model": search.best_estimator_,
            "params": search.best_params_,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cv_score": -search.best_score_,
        }

        print(f"{model_name} Results:")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  CV Score: {-search.best_score_:.4f}")

        if rmse < best_score:
            best_score = rmse
            best_model = search.best_estimator_
            best_model_name = model_name

    # ======================================================
    # 6. SAVE BEST MODEL AND RESULTS
    # ======================================================
    save_dir = os.path.join(base_dir, country.capitalize())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the best model
    joblib.dump(best_model, f"{save_dir}/{country.lower()}_improved_model.pkl")
    joblib.dump(scaler, f"{save_dir}/{country.lower()}_scaler.pkl")
    joblib.dump(list(X.columns), f"{save_dir}/{country.lower()}_improved_features.pkl")

    # Save comprehensive results
    best_result = results[best_model_name]
    report = f"""
IMPROVED MODEL PERFORMANCE REPORT - {country}
{'='*60}

BEST MODEL: {best_model_name}

PERFORMANCE METRICS:
RMSE: {best_result['rmse']:,.2f}
MAE:  {best_result['mae']:,.2f}
R²:   {best_result['r2']:.4f}
CV Score: {best_result['cv_score']:.4f}

BEST PARAMETERS:
{best_result['params']}

DATASET INFO:
Original shape: {df.shape}
Final features: {len(X.columns)}
Training samples: {len(X_train)}
Test samples: {len(X_test)}

ALL MODEL COMPARISON:
"""

    for name, res in results.items():
        report += f"\n{name}:"
        report += f"\n  RMSE: {res['rmse']:,.2f}"
        report += f"\n  MAE:  {res['mae']:,.2f}"
        report += f"\n  R²:   {res['r2']:.4f}"
        report += f"\n  CV:   {res['cv_score']:.4f}\n"

    with open(f"{save_dir}/{country.lower()}_improved_metrics.txt", "w") as f:
        f.write(report)

    print(f"\n{report}")

    # ======================================================
    # 7. FEATURE IMPORTANCE VISUALIZATION
    # ======================================================
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(12, 8))
        importance = best_model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]

        plt.barh(range(20), importance[indices])
        plt.yticks(range(20), [X.columns[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.title(f"Top 20 Feature Importances - {best_model_name} ({country})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/improved_feature_importance.png", dpi=200, bbox_inches="tight"
        )
        plt.close()

    return results


if __name__ == "__main__":
    countries = ["iraq", "lebanon", "usa"]
    all_results = {}

    for country in countries:
        try:
            results = train_improved_model(country)
            if results:
                all_results[country] = results
        except Exception as e:
            print(f"Error processing {country}: {e}")
            import traceback

            traceback.print_exc()

    # Summary comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - IMPROVED MODELS")
    print(f"{'='*80}")

    for country, results in all_results.items():
        print(f"\n{country.upper()}:")
        best_model = min(results.items(), key=lambda x: x[1]["rmse"])
        best_name, best_metrics = best_model
        print(f"  Best Model: {best_name}")
        print(f"  RMSE: {best_metrics['rmse']:,.2f}")
        print(f"  MAE:  {best_metrics['mae']:,.2f}")
        print(f"  R²:   {best_metrics['r2']:.4f}")
