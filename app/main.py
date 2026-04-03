from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np

from app.model_loader import load_model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model, metadata = load_model()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    city: str = Form(...),
    age: int = Form(...),
    exposure_minutes: float = Form(...),
    activity: str = Form(...),
    mask: str = Form(...),
    asthma: str = Form(...),
    smoker: str = Form(...),
):

    # Example feature vector (same order used in training)
    X = np.array([[age, exposure_minutes]])

    risk = model.predict_proba(X)[0][1]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "risk": round(float(risk), 3),
            "city": city
        },
    )
