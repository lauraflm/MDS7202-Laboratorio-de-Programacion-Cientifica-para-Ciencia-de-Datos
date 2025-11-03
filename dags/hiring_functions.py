# dags/hiring_functions.py
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import gradio as gr

# Base de trabajo: AIRFLOW_HOME/runs/<ds>/{raw,splits,models}
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
RUNS_DIR = os.path.join(AIRFLOW_HOME, "runs")
_dir = lambda ds: os.path.join(RUNS_DIR, ds)

# 1) (3 pts) Crea carpeta por fecha de ejecución + subcarpetas raw/splits/models
def create_folders(**kwargs):
    ds = kwargs["ds"]                 # Airflow inyecta 'ds' (YYYY-MM-DD)
    base = _dir(ds)
    os.makedirs(os.path.join(base, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base, "splits"), exist_ok=True)
    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    print(f"[OK] carpetas listas en {base}")

# 2) (3 pts) Lee raw/data_1.csv, hace hold-out 80/20 estratificado y guarda en splits/
def split_data(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    data_path = os.path.join(base, "raw", "data_1.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No existe {data_path}")

    df = pd.read_csv(data_path)
    if "HiringDecision" not in df.columns:
        raise ValueError("Falta columna objetivo 'HiringDecision'")

    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    splits = os.path.join(base, "splits")
    X_train.to_csv(os.path.join(splits, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(splits, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(splits, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(splits, "y_test.csv"), index=False)
    print("[OK] split 80/20 guardado en splits/")

# 3) (8 pts) Preprocesa + entrena RandomForest, guarda .joblib y PRINT de accuracy y f1(positivo=1)
def preprocess_and_train(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    splits = os.path.join(base, "splits")
    models = os.path.join(base, "models")

    X_train = pd.read_csv(os.path.join(splits, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(splits, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(splits, "y_train.csv")).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(splits, "y_test.csv")).squeeze("columns")

    # Categóricas esperadas del lab
    cat_cols = X_train.columns.intersection(["Gender", "EducationLevel", "RecruitmentStrategy"])
    num_cols = X_train.columns.difference(cat_cols)


    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1p = f1_score(y_test, preds, pos_label=1)

    out = os.path.join(models, "model_random_forest.joblib")
    joblib.dump(pipe, out)

    print(f"[METRICAS] accuracy={acc:.4f} | f1(positivo=1)={f1p:.4f}")
    print(f"[OK] modelo guardado en {out}")

# 4) (1 pt) Interfaz Gradio
def predict(file, model_path):
    pipeline = joblib.load(model_path)
    # Gradio entrega un objeto con .name; soportamos ambas formas
    file_path = getattr(file, "name", file)
    input_data = pd.read_json(file_path)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = pd.Series(predictions).map({0: "No contratado", 1: "Contratado"})
    labels = labels.tolist()
    return {'Predicción': labels[0]}

def gradio_interface(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    model_path = os.path.join(base, "models", "model_random_forest.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en {model_path}. Entrena antes de lanzar la UI.")

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    # Si task de Airflow no bloqueado ->> prevent_thread_lock=True.
    interface.launch(server_name="0.0.0.0", share=True, prevent_thread_lock=True)