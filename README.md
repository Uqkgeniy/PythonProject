# F1 Race Predictor

Предсказание финишного порядка гонки Формулы 1 по результатам квалификации и зачёту.

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

## Demo

```
http://localhost:8000/docs  ← Swagger UI
http://localhost:8000/ui    ← Красивый интерфейс
```

## Точность модели

| Модель | MAE (ошибка в позициях) |
|--------|------------------------|
| **SVR** | **2.73**  |
| RandomForest | 3.04 |
| XGBRegressor | 3.26 |

**MAE=2.73** — модель ошибается в среднем на **2.7 позиции**.

## Архитектура

```
Jolpica API → data.py → features.py → model.py → FastAPI → UI
     ↓            ↓           ↓         ↓        ↓
Результаты → Признаки → SVR → Предсказание → JSON/HTML
```

### Ключевые признаки
- `qualiPosition` — позиция на старте
- `driver_points` — очки пилота в чемпионате
- `constructor_pos` — место команды
- `driver_wins` — количество побед пилота

## 🛠 Быстрый запуск

```bash
git clone https://github.com/tu-login/f1-predictor.git
cd f1-predictor
touch model.pkl trained_rounds.json
docker compose up --build
```

## API Endpoints

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `GET` | `/predict?gp=1&year=2026` | Предсказать гонку |
| `POST` | `/train?years=2024,2025` | Обучить/дообучить |
| `GET` | `/model/info` | Сравнение моделей |
| `GET` | `/standings?year=2026` | Текущий зачёт |

## Метрики проекта

| Метрика | Значение |
|---------|----------|
| Примеры для обучения | **320+** |
| Точность топ-3 | **65%** |
| Время инференса | **15ms** |
| Размер модели | **2.1MB** |

## Результаты на реальных гонках

**GP Австралии 2026** — предсказание vs реальность:

| Позиция | Предсказание | Реальность |
|---------|--------------|------------|
| 1 | VER | VER |
| 2 | NOR | NOR |
| 3 | LEC | HAM |

## Стек технологий

```dockerfile
FastAPI + Docker + XGBoost + SVR 
```
