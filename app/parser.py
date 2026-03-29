import requests
import pandas as pd
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "http://api.jolpi.ca/ergast/f1"


def get_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_historical_data(start_year=1950, end_year=2023):
    all_race_data = []
    session = get_session()

    for year in range(start_year, end_year + 1):
        races_url = f"{BASE_URL}/{year}.json"

        try:
            response = session.get(races_url, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Пропуск {year} года из-за ошибки сети: {e}")
            time.sleep(5)
            continue

        races = response.json().get('MRData', {}).get('RaceTable', {}).get('Races', [])

        for race in races:
            round_num = race.get('round')
            race_name = race.get('raceName')

            results_url = f"{BASE_URL}/{year}/{round_num}/results.json"
            qual_url = f"{BASE_URL}/{year}/{round_num}/qualifying.json"

            try:
                res_response = session.get(results_url, timeout=60)
                qual_response = session.get(qual_url, timeout=60)

                if res_response.status_code == 200 and qual_response.status_code == 200:
                    race_results = res_response.json().get('MRData', {}).get('RaceTable', {}).get('Races', [])
                    qual_results = qual_response.json().get('MRData', {}).get('RaceTable', {}).get('Races', [])

                    if race_results and qual_results:
                        all_race_data.append({
                            "year": year,
                            "round": round_num,
                            "race_name": race_name,
                            "results": race_results[0].get('Results', []) if race_results else [],
                            "qualifying": qual_results[0].get('QualifyingResults', []) if qual_results else []
                        })
                        print(f"  Данные для {race_name} ({year}) успешно загружены.")

            except requests.exceptions.RequestException as e:
                print(f"Ошибка при загрузке этапа {round_num} ({year}): {e}")

            time.sleep(0.5)

        print(f"Год {year} успешно выгружен.")
        time.sleep(1)

    df = pd.DataFrame(all_race_data)
    return df


if __name__ == "__main__":
    print("Начало загрузки данных...")
    df_expanded = fetch_historical_data(start_year=2022, end_year=2025)
    df_expanded.to_csv("f1_expanded_dataset.csv", index=False)
    print("Загрузка завершена. Файл сохранен.")