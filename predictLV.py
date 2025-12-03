import logging
import fastf1
import pandas as pd
import os
from xgboost import XGBRegressor

CACHE_DIR = 'cache'
TARGET_GP = 23
TRAIN_GPS = [20, 21, 22]
YEAR = 2025
fastf1.set_log_level(logging.ERROR)

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

team_points = {
    "McLaren": 800,
    "Ferrari": 382,
    "Red Bull": 426,
    "Mercedes": 459,
    "Aston Martin": 80,
    "Alpine": 22,
    "Haas": 73,
    "Racing Bulls": 92,
    "Williams": 137,
    "Sauber": 68
}
max_points = max(team_points.values())

driver_to_team = {
    "VER": "Red Bull", "TSU": "Red Bull",
    "LAW": "Racing Bulls", "HAD": "Racing Bulls",
    "NOR": "McLaren", "PIA": "McLaren",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "SAI": "Williams", "ALB": "Williams",
    "GAS": "Alpine", "COL": "Alpine",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "HUL": "Sauber", "BOR": "Sauber",
    "OCO": "Haas", "BEA": "Haas"
}

def get_data_for_gp(year, gp_num):
    print(f"\n--- Сбор данных: Гран-при #{gp_num} ---")

    try:
        session_q = fastf1.get_session(year, gp_num, "Q")
        session_q.load(weather=False, telemetry=False, messages=False)

        grid = session_q.results[['Abbreviation', 'Position']].copy()
        grid.columns = ['Driver', 'GridPosition']

        fastest_lap = session_q.laps.pick_fastest()
        if fastest_lap is not None and not pd.isna(fastest_lap['LapTime']):
            pole_time = fastest_lap['LapTime'].total_seconds()
        else:
            pole_time = 80.0

        qualy_deltas = []

        for drv in grid['Driver']:
            delta = 4.0
            try:
                drv_laps = session_q.laps.pick_driver(drv)

                if len(drv_laps) > 0:
                    best_lap = drv_laps.pick_fastest()
                    if best_lap is not None and not pd.isna(best_lap['LapTime']):
                        delta = best_lap['LapTime'].total_seconds() - pole_time
            except Exception:
                pass

            qualy_deltas.append({'Driver': drv, 'QualyDelta': delta})

        grid = grid.merge(pd.DataFrame(qualy_deltas), on='Driver')

    except Exception as e:
        print(f"  [!] Ошибка загрузки квалификации: {e}")
        return None

    paces = []
    for fp in ['FP1', 'FP2', 'FP3']:
        try:
            session_fp = fastf1.get_session(year, gp_num, fp)
            session_fp.load(weather=False, telemetry=False, messages=False)

            laps = session_fp.laps.pick_accurate()
            if len(laps) > 0:
                laps = laps.copy()
                laps['Seconds'] = laps['LapTime'].dt.total_seconds()
                paces.append(laps[['Driver', 'Seconds']])
        except:
            continue

    if paces:
        all_fp = pd.concat(paces)
        fp_pace = all_fp.groupby('Driver')['Seconds'].median().reset_index(name='PracticePace')
        best_pace = fp_pace['PracticePace'].min()
        fp_pace['PracticePaceDelta'] = fp_pace['PracticePace'] - best_pace
    else:
        fp_pace = pd.DataFrame({'Driver': grid['Driver'], 'PracticePaceDelta': 1.0})

    try:
        session_r = fastf1.get_session(year, gp_num, "R")
        session_r.load(weather=False, telemetry=False, messages=False)
        results = session_r.results[['Abbreviation', 'Position']].copy()
        results.columns = ['Driver', 'ActualPosition']
    except Exception as e:
        print(f"  [!] Нет данных гонки: {e}")
        return None

    df = grid.merge(fp_pace[['Driver', 'PracticePaceDelta']], on='Driver', how='left')
    df = df.merge(results, on='Driver', how='inner')

    df['Team'] = df['Driver'].map(driver_to_team)
    df['TeamScore'] = df['Team'].map(team_points) / max_points

    df['PracticePaceDelta'] = df['PracticePaceDelta'].fillna(2.0)
    df['QualyDelta'] = df['QualyDelta'].fillna(4.0)

    return df

try:
    print(">>> Сбор обучающей выборки (TRAIN)...")
    train_dfs = []
    for gp in TRAIN_GPS:
        data = get_data_for_gp(YEAR, gp)
        if data is not None:
            train_dfs.append(data)

    if not train_dfs:
        raise ValueError("Нет данных для обучения!")

    train_data = pd.concat(train_dfs)
    print(f"Обучающая выборка: {len(train_data)} строк")

    print("\n>>> Сбор тестовой выборки (TEST)...")
    test_data = get_data_for_gp(YEAR, TARGET_GP)
    if test_data is None:
        raise ValueError("Нет данных для теста!")

    features = ['GridPosition', 'QualyDelta', 'PracticePaceDelta', 'TeamScore']

    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(train_data[features], train_data['ActualPosition'])

    test_data['PredictedScore'] = model.predict(test_data[features])
    test_data = test_data.sort_values('PredictedScore')
    test_data['PredictedPosition'] = range(1, len(test_data) + 1)

    test_data['Error'] = abs(test_data['ActualPosition'] - test_data['PredictedPosition'])

    print(f"\n=== ПРОГНОЗ GP #{TARGET_GP} (Катар) ===")
    print(test_data[['Driver', 'Team', 'GridPosition', 'PredictedPosition', 'ActualPosition', 'Error']].to_string(
        index=False))
    print(f"\nMAE: {test_data['Error'].mean():.2f}")

except Exception as e:
    print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
