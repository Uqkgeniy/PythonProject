import json
import pickle
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings

from app.features import build_features, FEATURE_COLS

MODEL_PATH = Path("/app/model.pkl") if Path("/app").exists() else Path("model.pkl")
COMPARISON_PATH = MODEL_PATH.parent / "model_comparison.json"
warnings.filterwarnings("ignore", category=UserWarning)


def _get_candidates() -> dict:
    return {
        "XGBRegressor": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.5))
        ])
    }


def train_model(years: list[int] = [2023, 2024, 2025]) -> tuple:
    all_data = []

    for year in years:
        for round_num in range(1, 25):
            try:
                df = build_features(year, round_num, include_result=True)
                if df.empty or "race_position" not in df.columns:
                    continue
                df = df.dropna(subset=["race_position"])
                if len(df) < 10:
                    continue
                all_data.append(df)
                print(f"done {year} R{round_num}: {len(df)} гонщиков")
            except Exception as e:
                print(f"none {year} R{round_num}: {e}")

    if not all_data:
        raise ValueError("Нет данных для обучения")

    full_df = pd.concat(all_data, ignore_index=True)
    X = full_df[FEATURE_COLS].fillna(0)
    y = full_df["race_position"]
    print(f"\nВсего примеров: {len(full_df)} ({len(all_data)} гонок)")

    candidates = _get_candidates()
    comparison = {}
    best_name, best_score, best_model = None, float("inf"), None

    print("\nСравнение моделей (5-fold CV):")
    for name, model in candidates.items():
        scores = cross_val_score(
            model, X, y,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        mae = float(-scores.mean())
        std = float(scores.std())
        comparison[name] = {"MAE": round(mae, 3), "STD": round(std, 3)}
        print(f"  {name:20s} MAE={mae:.3f} ± {std:.3f}")

        if mae < best_score:
            best_score = mae
            best_name = name
            best_model = model

    print(f"\nПобедитель: {best_name} (MAE={best_score:.3f})")

    best_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)

    comparison["winner"] = best_name
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Модель сохранена: {MODEL_PATH}")
    return best_model, comparison


def load_model() -> tuple:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Модель не обучена. Вызови POST /train")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["model"], data["name"]
    return data, "Unknown"


def predict_race(year: int, round_num: int) -> list[dict]:
    model, model_name = load_model()

    df = build_features(year, round_num, include_result=False)
    if df.empty:
        raise ValueError(f"Нет данных квалификации: {year} R{round_num}")

    X = df[FEATURE_COLS].fillna(0)
    df = df.copy()
    df["predicted_score"] = model.predict(X)
    df["predicted_position"] = (
        df["predicted_score"]
        .rank(method="first")
        .astype(int)
    )

    return (
        df.sort_values("predicted_position")[[
            "predicted_position",
            "code",
            "constructorId",
            "qualiPosition",
            "driver_points",
            "constructor_points",
        ]]
        .assign(model_used=model_name)
        .to_dict(orient="records")
    )
