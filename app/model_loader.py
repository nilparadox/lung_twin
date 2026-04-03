import os
import joblib

MODEL_PATH = "models/digital_lung_model_compact_v2.joblib"
META_PATH = "models/digital_lung_model_compact_v2_metadata.joblib"

model = None
metadata = None


def load_model():
    global model, metadata

    if model is not None:
        return model, metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if os.path.exists(META_PATH):
        metadata = joblib.load(META_PATH)

    return model, metadata
