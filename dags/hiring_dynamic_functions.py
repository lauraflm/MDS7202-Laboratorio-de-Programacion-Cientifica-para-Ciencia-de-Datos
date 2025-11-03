# dags/hiring_dynamic_functions.py
import os
import glob
import json
import joblib
import pandas as pd

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
RUNS_DIR = os.path.join(AIRFLOW_HOME, "runs")
_dir = lambda ds: os.path.join(RUNS_DIR, ds)

# 1) crea carpetas por fecha de ejecución
def create_folders(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    for sub in ("raw", "preprocessed", "splits", "models"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    print(f"[OK] carpetas listas en {base}")

# 2) carga 1 o 2 CSV que hayan llegado a raw/ y los concatena
def load_and_merge(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    raw = os.path.join(base, "raw")
    out = os.path.join(base, "preprocessed", "merged.csv")

    # admite data_1.csv y data_2.csv si existe
    paths = [p for p in (os.path.join(raw, "data_1.csv"), os.path.join(raw, "data_2.csv")) if os.path.exists(p)]
    if len(paths) == 0:
        raise FileNotFoundError("No hay archivos en raw/ (esperado: data_1.csv y/o data_2.csv).")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[OK] merge → {out} (n={len(df)})")

# 3) hold-out 80/20 con estratificación
def split_data(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    merged = os.path.join(base, "preprocessed", "merged.csv")
    splits = os.path.join(base, "splits")

    df = pd.read_csv(merged)
    if "HiringDecision" not in df.columns:
        raise ValueError("Falta columna objetivo 'HiringDecision'")

    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train.to_csv(os.path.join(splits, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(splits, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(splits, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(splits, "y_test.csv"), index=False)
    print("[OK] split guardado en splits/")

# ——— util común
def _make_preproc(X: pd.DataFrame):
    cat_cols = [c for c in ["Gender", "EducationLevel", "RecruitmentStrategy"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
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
    return pre

# 4) entrena UN modelo (se llamará 3 veces en paralelo desde el DAG)
def train_model(model_name: str, **kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    splits = os.path.join(base, "splits")
    models = os.path.join(base, "models")

    X_train = pd.read_csv(os.path.join(splits, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(splits, "y_train.csv")).squeeze("columns")

    pre = _make_preproc(X_train)

    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    elif model_name == "logreg":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "gboost":
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    out = os.path.join(models, f"model_{model_name}.joblib")
    joblib.dump(pipe, out)
    print(f"[OK] {model_name} entrenado → {out}")

# 5) evalúa TODOS los modelos entrenados y deja el mejor como best_model.joblib
def evaluate_and_select(**kwargs):
    ds = kwargs["ds"]
    base = _dir(ds)
    splits = os.path.join(base, "splits")
    models = os.path.join(base, "models")

    X_test = pd.read_csv(os.path.join(splits, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(splits, "y_test.csv")).squeeze("columns")

    candidates = sorted(glob.glob(os.path.join(models, "model_*.joblib")))
    if len(candidates) == 0:
        raise FileNotFoundError("No hay modelos en models/ para evaluar.")

    best_path, best_acc = None, -1.0
    for path in candidates:
        pipe = joblib.load(path)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"[ACC] {os.path.basename(path)} = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_path = acc, path

    final = os.path.join(models, "best_model.joblib")
    # reemplazar/renombrar al mejor (copy -> overwrite)
    import shutil
    shutil.copyfile(best_path, final)
    print(f"[MEJOR] {os.path.basename(best_path)} con accuracy={best_acc:.4f} → {final}")

# 6) interfaz gradio que usa el MEJOR modelo
def gradio_best_interface(**kwargs):
    import gradio as gr
    ds = kwargs["ds"]
    base = _dir(ds)
    model_path = os.path.join(base, "models", "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe {model_path}. Ejecuta evaluate_and_select primero.")

    def _predict(file):
        path = getattr(file, "name", file)
        df = pd.read_json(path)
        pipe = joblib.load(model_path)
        y = pipe.predict(df)
        etiqueta = "Contratado" if int(y[0]) == 1 else "No contratado"
        return {"Predicción": etiqueta}

    iface = gr.Interface(
        fn=_predict,
        inputs=gr.File(label="Sube JSON"),
        outputs="json",
        title="Hiring Decision (Mejor Modelo Automático)",
        description="El pipeline entrena varios modelos y selecciona el mejor; esta UI usa ese modelo."
    )
    # para que el task termine y la app siga corriendo:
    info = iface.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=True)
    share_url = getattr(info, "share_url", None) if info else None
    print(f"[GRADIO] http://0.0.0.0:7860 | público: {share_url or '(ver salida de Gradio)'}")
    return {"share_url": share_url}

