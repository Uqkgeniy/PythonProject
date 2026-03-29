import pandas as pd
import numpy as np
from app.data import (
    get_driver_standings,
    get_constructor_standings,
    get_qualifying_results,
    get_race_results,
    get_practice_telemetry,
    get_sprint_results,
    get_weather_data
)

FEATURE_COLS = [
    'qualiPosition',
    'driver_points',
    'driver_champ_pos',
    'driver_wins',
    'constructor_points',
    'constructor_pos',
    'sprint_points',
    'sprint_position',
    'track_temp',
    'is_rain',
    'avg_speed',
    'max_speed',
    'throttle_pct',
    'brake_pct',
    'drs_pct'
]


def build_features(year: int, round_num: int, include_result: bool = True) -> pd.DataFrame:
    df = get_qualifying_results(year, round_num)
    if df.empty:
        return pd.DataFrame()

    driver_st = get_driver_standings(year, round_num)
    if not driver_st.empty:
        df = df.merge(driver_st, on='code', how='left')
    else:
        df['driver_points'] = 0.0
        df['driver_champ_pos'] = 20
        df['driver_wins'] = 0

    constr_st = get_constructor_standings(year, round_num)
    if not constr_st.empty:
        df = df.merge(constr_st, on='constructorId', how='left')
    else:
        df['constructor_points'] = 0.0
        df['constructor_pos'] = 10

    sprints = get_sprint_results(year, round_num)
    if not sprints.empty:
        df = df.merge(sprints, on='code', how='left')
        df['sprint_points'] = df['sprint_points'].fillna(0.0)
        df['sprint_position'] = df['sprint_position'].fillna(20)
    else:
        df['sprint_points'] = 0.0
        df['sprint_position'] = 20

    telemetry = get_practice_telemetry(year, round_num)
    if not telemetry.empty:
        telemetry = telemetry.rename(columns={'driver': 'code'})
        df = df.merge(telemetry, on='code', how='left')
    else:
        for col in ['avg_speed', 'max_speed', 'throttle_pct', 'brake_pct', 'drs_pct']:
            df[col] = 0.0

    weather = get_weather_data(year, round_num)
    if not weather.empty:
        df['track_temp'] = weather['track_temperature'].iloc[0]
        df['is_rain'] = weather['rainfall'].iloc[0]
    else:
        df['track_temp'] = 30.0
        df['is_rain'] = 0

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'pos' in col.lower() or 'position' in col.lower():
            df[col] = df[col].fillna(20)
        else:
            df[col] = df[col].fillna(0.0)

    df['year'] = year
    df['round'] = round_num

    if include_result:
        results = get_race_results(year, round_num)
        if not results.empty:
            df = df.merge(results, on='code', how='left')
            df['race_position'] = df['race_position'].fillna(20)

    return df