# model_utils.py
import os
import mlflow


def load_model():
    model_uri = os.getenv("MODEL_URI", "models:/sodai_model/1")
    model = mlflow.sklearn.load_model(model_uri)
    return model
