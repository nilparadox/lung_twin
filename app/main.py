from __future__ import annotations

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.env_loader import load_environment
from app.model_predict import predict_with_trained_model
from app.plots import generate_pollution_plot, generate_risk_plot
from app.recommendations import generate_recommendations
from app.services import geocode_place, get_live_air, get_weather

load_environment()

app = FastAPI(title="ZetaQ AirWise AI")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def default_form_data() -> dict:
    return {
        "city": "",
        "age": 30,
        "exposure_min": 45,
        "activity": "walk",
        "asthma": 0,
        "smoker": 0,
        "mask_type": "none",
    }


def build_chart_data(result: dict | None, form_data: dict, air_data: dict | None):
    if not result:
        return None

    exposure_min = float(form_data["exposure_min"])
    safe_minutes = float(result["safe_minutes"])
    recovery_minutes = float(result["recovery_minutes"])

    chart = {
        "risk_pct": float(result["risk_score"]),
        "exposure_pct": min(100.0, (exposure_min / max(safe_minutes, 1.0)) * 100.0),
        "safe_pct": min(100.0, safe_minutes / 240.0 * 100.0),
        "recovery_pct": min(100.0, recovery_minutes / 1440.0 * 100.0),
        "lung_pct": min(100.0, float(result["lung_load"]) * 18.0),
        "inflam_pct": min(100.0, float(result["inflammation_score"]) * 55.0),
        "irrit_pct": min(100.0, float(result["irritation_probability"]) * 100.0),
        "oxygen_pct": min(100.0, float(result["oxygen_drop_pct"]) * 12.5),
    }

    if air_data:
        chart["pm25_pct"] = min(100.0, float(air_data["pm25"]) / 2.5)
        chart["pm10_pct"] = min(100.0, float(air_data["pm10"]) / 4.0)
        chart["no2_pct"] = min(100.0, float(air_data["no2"]) / 1.5)
        chart["o3_pct"] = min(100.0, float(air_data["o3"]) / 2.0)

    return chart


def render_page(
    request: Request,
    result=None,
    air_data=None,
    weather=None,
    place=None,
    form_data=None,
    error=None,
    risk_plot=None,
    pollution_plot=None,
    recommendations=None,
):
    if form_data is None:
        form_data = default_form_data()

    chart_data = build_chart_data(result, form_data, air_data)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "air_data": air_data,
            "weather": weather,
            "place": place,
            "form_data": form_data,
            "chart_data": chart_data,
            "error": error,
            "risk_plot": risk_plot,
            "pollution_plot": pollution_plot,
            "recommendations": recommendations or [],
        },
    )


@app.head("/")
def head_ok():
    return Response(status_code=200)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_page(request)


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    city: str = Form(...),
    age: int = Form(...),
    exposure_min: float = Form(...),
    activity: str = Form(...),
    asthma: int = Form(...),
    smoker: int = Form(...),
    mask_type: str = Form(...),
):
    form_data = {
        "city": city,
        "age": age,
        "exposure_min": exposure_min,
        "activity": activity,
        "asthma": asthma,
        "smoker": smoker,
        "mask_type": mask_type,
    }

    try:
        place = geocode_place(city)
        air_data = get_live_air(place["latitude"], place["longitude"])
        weather = get_weather(place["latitude"], place["longitude"])

        result = predict_with_trained_model(
            age=age,
            pm25=air_data["pm25"],
            pm10=air_data["pm10"],
            temp_c=weather["temp_c"],
            humidity=weather["humidity"],
            exposure_min=exposure_min,
            activity=activity,
            asthma=asthma,
            smoker=smoker,
            mask_type=mask_type,
        )

        if weather.get("weather_note"):
            result["advice"] += f" Note: {weather['weather_note']}"

        risk_plot = generate_risk_plot(result["risk_score"])
        pollution_plot = generate_pollution_plot(air_data["pm25"], air_data["pm10"])
        recommendations = generate_recommendations(result, form_data, air_data)

        return render_page(
            request,
            result=result,
            air_data=air_data,
            weather=weather,
            place=place,
            form_data=form_data,
            risk_plot=risk_plot,
            pollution_plot=pollution_plot,
            recommendations=recommendations,
        )
    except Exception as e:
        return render_page(request, form_data=form_data, error=str(e))


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
