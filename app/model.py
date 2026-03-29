import json
import pickle
import pandas as pd
from pathlib import Path
from xgboost import XGBRanker
import warnings

from app.features import build_features, FEATURE_COLS

MODEL_PATH = Path("/app/model.pkl") if Path("/app").exists() else Path("model.pkl")
COMPARISON_PATH = MODEL_PATH.parent / "model_comparison.json"
warnings.filterwarnings("ignore", category=UserWarning)


def train_model(years: list[int] = [2024, 2025]) -> tuple:
    all_data = []

    for year in years:
        for round_num in range(1, 25):
            print(f"[{year}] Скачиваю данные для этапа {round_num}...")
            try:
                df = build_features(year, round_num, include_result=True)
                if df.empty or "race_position" not in df.columns:
                    print(f"  -> Пропуск: нет данных гонки")
                    continue
                df = df.dropna(subset=["race_position"])
                if len(df) < 10:
                    print(f"  -> Пропуск: слишком мало данных")
                    continue
                all_data.append(df)
                print(f"  -> Успех: собрано {len(df)} строк")
            except Exception as e:
                print(f"  -> Ошибка: {e}")

    if not all_data:
        raise ValueError("Нет данных для обучения")

    print("\nНачинаю обучение XGBRanker...")
    full_df = pd.concat(all_data, ignore_index=True)

    full_df["qid"] = full_df["year"] * 100 + full_df["round"]
    full_df = full_df.sort_values(["qid", "race_position"])

    X = full_df[FEATURE_COLS].fillna(0)

    y = 21 - full_df["race_position"]
    y = y.clip(lower=0)

    model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    model.fit(X, y, qid=full_df["qid"])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "name": "XGBRanker"}, f)

    comparison = {
        "XGBRanker": {"Status": "Trained", "Races": int(full_df["qid"].nunique())},
        "winner": "XGBRanker"
    }

    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    print("Обучение завершено. Модель сохранена!")
    return model, comparison

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
        raise ValueError(f"Нет данных: {year} R{round_num}")

    X = df[FEATURE_COLS].fillna(0)
    df = df.copy()

    df["predicted_score"] = model.predict(X)

    df["predicted_position"] = (
        df["predicted_score"]
        .rank(method="first", ascending=False)
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