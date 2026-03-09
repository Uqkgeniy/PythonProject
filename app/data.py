import fastf1
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from pathlib import Path
import os

if Path("/app").exists() and os.access("/app", os.W_OK):
    CACHE_DIR = Path("/app/cache")
else:
    CACHE_DIR = Path(__file__).parent.parent / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

def _jolpica_get(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json().get("MRData", {})
    except Exception as e:
        print(f"[Jolpica] Ошибка запроса {url}: {e}")
        return {}


def get_current_drivers() -> pd.DataFrame:
    data = _jolpica_get(f"{JOLPICA_BASE}/current/drivers.json?limit=100")
    drivers = data.get("DriverTable", {}).get("Drivers", [])

    if not drivers:
        return pd.DataFrame(columns=["driverId", "code", "name", "number"])

    return pd.DataFrame([{
        "driverId": d["driverId"],
        "code":     d.get("code", d["driverId"][:3].upper()),
        "name":     f"{d['givenName']} {d['familyName']}",
        "number":   int(d.get("permanentNumber", 0))
    } for d in drivers])


def get_driver_standings(year: int, round_num: int) -> pd.DataFrame:
    if round_num <= 1:
        drivers = get_current_drivers()
        drivers["driver_points"]    = 0.0
        drivers["driver_champ_pos"] = range(1, len(drivers) + 1)
        drivers["driver_wins"]      = 0
        return drivers[["code", "driver_points",
                         "driver_champ_pos", "driver_wins"]]

    url = f"{JOLPICA_BASE}/{year}/{round_num - 1}/driverStandings.json"
    data = _jolpica_get(url)
    standings_lists = data.get("StandingsTable", {}).get("StandingsLists", [])

    if not standings_lists:
        return pd.DataFrame(columns=["code", "driver_points",
                                      "driver_champ_pos", "driver_wins"])

    standings = standings_lists[0]["DriverStandings"]
    return pd.DataFrame([{
        "code":             s["Driver"].get("code", ""),
        "driver_points":    float(s["points"]),
        "driver_champ_pos": int(s["position"]),
        "driver_wins":      int(s["wins"])
    } for s in standings])


def get_constructor_standings(year: int, round_num: int) -> pd.DataFrame:
    if round_num <= 1:
        return pd.DataFrame(columns=["constructorId",
                                      "constructor_points", "constructor_pos"])

    url = f"{JOLPICA_BASE}/{year}/{round_num - 1}/constructorStandings.json"
    data = _jolpica_get(url)
    standings_lists = data.get("StandingsTable", {}).get("StandingsLists", [])

    if not standings_lists:
        return pd.DataFrame(columns=["constructorId",
                                      "constructor_points", "constructor_pos"])

    standings = standings_lists[0]["ConstructorStandings"]
    return pd.DataFrame([{
        "constructorId":      s["Constructor"]["constructorId"],
        "constructor_points": float(s["points"]),
        "constructor_pos":    int(s["position"])
    } for s in standings])


def get_qualifying_results(year: int, round_num: int) -> pd.DataFrame:
    url = f"{JOLPICA_BASE}/{year}/{round_num}/qualifying.json"
    data = _jolpica_get(url)
    races = data.get("RaceTable", {}).get("Races", [])

    if not races:
        return pd.DataFrame(columns=["qualiPosition", "code", "constructorId"])

    results = races[0].get("QualifyingResults", [])
    return pd.DataFrame([{
        "qualiPosition": int(r["position"]),
        "code":          r["Driver"].get("code", ""),
        "constructorId": r["Constructor"]["constructorId"]
    } for r in results])


def get_race_results(year: int, round_num: int) -> pd.DataFrame:
    url = f"{JOLPICA_BASE}/{year}/{round_num}/results.json"
    data = _jolpica_get(url)
    races = data.get("RaceTable", {}).get("Races", [])

    if not races:
        return pd.DataFrame(columns=["code", "race_position"])

    results = races[0].get("Results", [])
    return pd.DataFrame([{
        "code":          r["Driver"].get("code", ""),
        "race_position": int(r["position"])
    } for r in results])


def _get_available_fp_sessions(year: int, round_num: int) -> list[str]:
    try:
        event = fastf1.get_event(year, round_num)
        fmt = event.get("EventFormat", "conventional")

        if fmt == "sprint_qualifying":
            return ["FP1"]
        elif fmt in ("sprint_shootout", "sprint"):
            return ["FP1", "FP2"]
        else:
            return ["FP1", "FP2", "FP3"]
    except Exception:
        return ["FP1", "FP2", "FP3"]


def _extract_telemetry_for_driver(session: fastf1.core.Session, driver: str) -> dict | None:
    try:
        driver_laps = (
            session.laps
            .pick_drivers(driver)
            .pick_accurate()
        )

        if driver_laps.empty:
            return None

        fastest = driver_laps.pick_fastest()
        if fastest is None or (hasattr(fastest, 'empty') and fastest.empty):
            return None

        drv_num = str(fastest.get("DriverNumber", ""))
        if drv_num not in session.car_data:
            return None

        car_data = fastest.get_car_data()
        if car_data is None or car_data.empty:
            return None

        required = {"Speed", "Throttle", "Brake", "DRS"}
        if not required.issubset(car_data.columns):
            return None

        brake_arr = car_data["Brake"].astype(bool)

        return {
            "driver":       driver,
            "avg_speed":    float(car_data["Speed"].mean()),
            "max_speed":    float(car_data["Speed"].max()),
            "throttle_pct": float((car_data["Throttle"] > 90).mean() * 100),
            "brake_pct":    float(brake_arr.mean() * 100),
            "drs_pct":      float((car_data["DRS"] >= 10).mean() * 100),
        }

    except Exception as e:
        print(f"[Telemetry] Ошибка {driver}: {e}")
        return None


def _get_openf1_session_key(year: int, round_num: int, session_type: str) -> int | None:

    url = f"{OPENF1_BASE}/sessions?year={year}&session_name={session_type}"
    data = _openf1_get(url)
    if not data:
        return None

    meetings_url = f"{OPENF1_BASE}/meetings?year={year}"
    meetings = _openf1_get(meetings_url)
    if not meetings:
        return None

    try:
        meetings_sorted = sorted(meetings, key=lambda m: m["date_start"])
        meeting = meetings_sorted[round_num - 1]
        meeting_key = meeting["meeting_key"]
    except (IndexError, KeyError):
        return None

    for s in data:
        if s.get("meeting_key") == meeting_key:
            return s["session_key"]
    return None


def _openf1_get(url: str) -> list:
    try:
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[OpenF1] Ошибка {url}: {e}")
        return []


def get_practice_telemetry(year: int, round_num: int) -> pd.DataFrame:
    fp_names = ["Practice 1", "Practice 2", "Practice 3"]
    records = []

    for fp_name in fp_names:
        session_key = _get_openf1_session_key(year, round_num, fp_name)
        if not session_key:
            print(f"[OpenF1] {fp_name}: сессия не найдена")
            continue

        drivers_url = f"{OPENF1_BASE}/drivers?session_key={session_key}"
        drivers = _openf1_get(drivers_url)
        if not drivers:
            continue

        for driver in drivers:
            driver_num = driver["driver_number"]
            code = driver.get("name_acronym", str(driver_num))

            tel_url = (f"{OPENF1_BASE}/car_data"
                       f"?session_key={session_key}"
                       f"&driver_number={driver_num}")
            tel_data = _openf1_get(tel_url)

            if not tel_data:
                continue

            tel_df = pd.DataFrame(tel_data)

            required = {"speed", "throttle", "brake", "drs"}
            if not required.issubset(tel_df.columns):
                continue

            records.append({
                "driver":       code,
                "fp":           fp_name,
                "avg_speed":    float(tel_df["speed"].mean()),
                "max_speed":    float(tel_df["speed"].max()),
                "throttle_pct": float((tel_df["throttle"] > 90).mean() * 100),
                "brake_pct":    float((tel_df["brake"] > 0).mean() * 100),
                "drs_pct":      float((tel_df["drs"] >= 10).mean() * 100),
            })

        print(f"[OpenF1] ✓ {fp_name} (key={session_key}): "
              f"{len(drivers)} гонщиков")

    if not records:
        print(f"[OpenF1] Нет данных для {year} R{round_num}")
        return pd.DataFrame(columns=[
            "driver", "avg_speed", "max_speed",
            "throttle_pct", "brake_pct", "drs_pct"
        ])

    df = pd.DataFrame(records)
    return (
        df.groupby("driver")
        .agg({
            "avg_speed":    "mean",
            "max_speed":    "mean",
            "throttle_pct": "mean",
            "brake_pct":    "mean",
            "drs_pct":      "mean",
        })
        .reset_index()
    )
