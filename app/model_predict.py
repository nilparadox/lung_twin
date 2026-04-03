from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


MODEL_PATH = Path("models/digital_lung_model_compact.joblib")
METADATA_PATH = Path("models/digital_lung_model_compact_metadata.joblib")

_MODEL = None
_METADATA = None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimate_baseline_lung(age: int, asthma: int, smoker: int, activity: str) -> float:
    b = 1.02

    if age < 18:
        b -= 0.08
    elif age < 35:
        b += 0.00
    elif age < 50:
        b -= 0.04
    elif age < 60:
        b -= 0.09
    elif age < 70:
        b -= 0.16
    else:
        b -= 0.24

    if asthma:
        b -= 0.14

    if smoker:
        b -= 0.11

    if activity == "exercise":
        b += 0.03
    elif activity == "jog":
        b += 0.01

    return round(clamp(b, 0.60, 1.08), 2)


def sensitivity_label(baseline_lung: float) -> str:
    if baseline_lung >= 0.97:
        return "low"
    if baseline_lung >= 0.82:
        return "moderate"
    return "high"


def load_model_bundle():
    global _MODEL, _METADATA

    if _MODEL is None or _METADATA is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        print(f"[MODEL] Loading compact trained model from {MODEL_PATH}")
        print(f"[MODEL] Loading compact metadata from {METADATA_PATH}")

        _MODEL = joblib.load(MODEL_PATH)
        _METADATA = joblib.load(METADATA_PATH)

    return _MODEL, _METADATA


def _age_adjustments(age: int):
    if age >= 75:
        return {
            "risk_bonus": 16.0,
            "safe_factor": 0.62,
            "recovery_factor": 1.45,
            "lung_factor": 1.24,
            "inflam_factor": 1.28,
            "oxygen_factor": 1.25,
        }
    if age >= 65:
        return {
            "risk_bonus": 11.0,
            "safe_factor": 0.72,
            "recovery_factor": 1.32,
            "lung_factor": 1.17,
            "inflam_factor": 1.20,
            "oxygen_factor": 1.18,
        }
    if age >= 55:
        return {
            "risk_bonus": 7.0,
            "safe_factor": 0.82,
            "recovery_factor": 1.20,
            "lung_factor": 1.10,
            "inflam_factor": 1.12,
            "oxygen_factor": 1.12,
        }
    if age >= 45:
        return {
            "risk_bonus": 3.5,
            "safe_factor": 0.91,
            "recovery_factor": 1.10,
            "lung_factor": 1.05,
            "inflam_factor": 1.06,
            "oxygen_factor": 1.06,
        }

    return {
        "risk_bonus": 0.0,
        "safe_factor": 1.00,
        "recovery_factor": 1.00,
        "lung_factor": 1.00,
        "inflam_factor": 1.00,
        "oxygen_factor": 1.00,
    }


def predict_with_trained_model(
    age: int,
    pm25: float,
    pm10: float,
    temp_c: float,
    humidity: float,
    exposure_min: float,
    activity: str,
    asthma: int,
    smoker: int,
    mask_type: str,
):
    model, metadata = load_model_bundle()
    features = metadata["features"]

    baseline_lung = estimate_baseline_lung(age, asthma, smoker, activity)

    feature_values = {
        "age": age,
        "pm25": pm25,
        "pm10": pm10,
        "temp_c": temp_c,
        "humidity": humidity,
        "exposure_min": exposure_min,
        "activity": activity,
        "asthma": asthma,
        "smoker": smoker,
        "mask_type": mask_type,
        "baseline_lung": baseline_lung,
    }

    X = pd.DataFrame([[feature_values[f] for f in features]], columns=features)
    pred = model.predict(X)[0]

    risk_score = clamp(float(pred[0]), 0.0, 100.0)
    safe_minutes = clamp(float(pred[1]), 5.0, 240.0)
    recovery_minutes = clamp(float(pred[2]), 5.0, 1440.0)
    lung_load = max(0.0, float(pred[3]))
    inflammation_score = max(0.0, float(pred[4]))
    irritation_probability = clamp(float(pred[5]), 0.0, 1.0)
    oxygen_drop_pct = clamp(float(pred[6]), 0.0, 12.0)

    adj = _age_adjustments(age)

    risk_score = clamp(risk_score + adj["risk_bonus"], 0.0, 100.0)
    safe_minutes = clamp(safe_minutes * adj["safe_factor"], 5.0, 240.0)
    recovery_minutes = clamp(recovery_minutes * adj["recovery_factor"], 5.0, 1440.0)
    lung_load = max(0.0, lung_load * adj["lung_factor"])
    inflammation_score = max(0.0, inflammation_score * adj["inflam_factor"])
    irritation_probability = clamp(irritation_probability * adj["inflam_factor"], 0.0, 1.0)
    oxygen_drop_pct = clamp(oxygen_drop_pct * adj["oxygen_factor"], 0.0, 12.0)

    if risk_score < 20:
        band = "low"
    elif risk_score < 40:
        band = "moderate"
    elif risk_score < 65:
        band = "high"
    else:
        band = "very high"

    if band == "low":
        advice = "Conditions look manageable for most healthy adults."
    elif band == "moderate":
        advice = "Air quality is not ideal, but routine outdoor work is generally manageable with some care."
    elif band == "high":
        advice = "Air pollution is elevated today. If you must stay outside, reduce exertion and take recovery breaks."
    else:
        advice = "Air pollution is quite heavy today. If outdoor work is necessary, use protection and take frequent recovery breaks."

    return {
        "risk_score": round(risk_score, 1),
        "risk_band": band,
        "safe_minutes": round(safe_minutes, 1),
        "recovery_minutes": round(recovery_minutes, 1),
        "lung_load": round(lung_load, 3),
        "inflammation_score": round(inflammation_score, 3),
        "irritation_probability": round(irritation_probability, 3),
        "oxygen_drop_pct": round(oxygen_drop_pct, 2),
        "baseline_lung": baseline_lung,
        "sensitivity": sensitivity_label(baseline_lung),
        "advice": advice,
    }
