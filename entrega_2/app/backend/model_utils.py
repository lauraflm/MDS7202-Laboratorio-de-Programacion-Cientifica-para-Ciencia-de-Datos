from pathlib import Path
import joblib
import sys

# Agregar el src local del backend
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_model():
    model_path = CURRENT_DIR / "models" / "xgb_best.pkl"
    return joblib.load(model_path)
