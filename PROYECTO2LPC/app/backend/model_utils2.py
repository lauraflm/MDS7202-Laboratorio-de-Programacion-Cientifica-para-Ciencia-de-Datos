# model_utils.py
import joblib
from pathlib import Path


def load_model():
    model_path = Path(__file__).parent / "model.pkl"
    model = joblib.load(model_path)
    return model
