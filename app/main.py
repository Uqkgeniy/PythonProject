import json
import datetime
import warnings
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


from app.model import train_model, predict_race, load_model, COMPARISON_PATH

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("F1 Predictor запущен")
    print("Документация: http://localhost:8000/docs")
    yield
    print("F1 Predictor остановлен")


app = FastAPI(
    title="F1 Race Predictor",
    description="Предсказание финишного порядка гонки Формулы 1",
    version="1.0.0",
    lifespan=lifespan
)

static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(str(static_path / "index.html"))


@app.get("/", tags=["Info"])
def root():
    return {
        "status": "ok",
        "endpoints": {
            "GET  /predict?gp=1&year=2026":        "Предсказание гонки",
            "GET  /results?gp=1&year=2026":        "Реальные результаты гонки",
            "GET  /standings?year=2026&round=1":   "Зачёт пилотов",
            "GET  /drivers":                        "Список гонщиков сезона",
            "GET  /model/info":                     "Сравнение моделей",
            "POST /train?years=2023,2024,2025":     "Обучить модель",
        }
    }


@app.get("/predict", tags=["Prediction"])
def predict(gp: int, year: int = None):
    if year is None:
        year = datetime.datetime.now().year
    try:
        result = predict_race(year, gp)
        return {
            "year":       year,
            "round":      gp,
            "model_used": result[0].get("model_used") if result else None,
            "prediction": result
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results", tags=["Data"])
def race_results(gp: int, year: int = None):
    if year is None:
        year = datetime.datetime.now().year
    try:
        from app.data import get_race_results
        df = get_race_results(year, gp)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Результаты {year} R{gp} недоступны (гонка ещё не состоялась?)"
            )
        return {
            "year":    year,
            "round":   gp,
            "results": df.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/standings", tags=["Data"])
def standings(year: int = None, round: int = 1):
    if year is None:
        year = datetime.datetime.now().year
    try:
        from app.data import get_driver_standings
        df = get_driver_standings(year, round)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Зачёт для {year} R{round} недоступен"
            )
        return {
            "year":      year,
            "before_round": round,
            "standings": df.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drivers", tags=["Data"])
def drivers():
    try:
        from app.data import get_current_drivers
        df = get_current_drivers()
        if df.empty:
            raise HTTPException(status_code=503, detail="Jolpica API недоступен")
        return {"count": len(df), "drivers": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
def model_info():
    if not COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Модель не обучена. Вызови POST /train"
        )
    with open(COMPARISON_PATH) as f:
        return json.load(f)


@app.post("/train", tags=["Model"])
def train(background_tasks: BackgroundTasks, years: str = "2023,2024,2025"):
    try:
        year_list = [int(y.strip()) for y in years.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="years должен быть списком через запятую: 2023,2024,2025"
        )

    background_tasks.add_task(train_model, year_list)
    return {
        "status":  "started",
        "message": f"Обучение запущено для сезонов: {year_list}",
        "tip":     "Статус смотри в /model/info после завершения"
    }


@app.get("/health", tags=["Info"])
def health():
    model_ready = False
    model_name = None
    try:
        _, model_name = load_model()
        model_ready = True
    except Exception:
        pass

    return {
        "status":      "ok",
        "model_ready": model_ready,
        "model_name":  model_name,
        "timestamp":   datetime.datetime.now().isoformat()
    }
