from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

# Ruta donde se montan las predicciones desde docker-compose
PREDICTIONS_DIR = Path("/predictions")

app = FastAPI(title="Recommender System SodAI")

class ClientRequest(BaseModel):
    customer_id: int


@app.get("/")
def root():
    return {"message": "Recsys backend OK"}


@app.post("/recommend")
def recommend(req: ClientRequest):
    try:
        # 1) Chequear que exista la carpeta
        if not PREDICTIONS_DIR.exists():
            return {"error": f"Predictions dir {PREDICTIONS_DIR} does not exist"}

        # 2) Buscar cualquier parquet de predicciones
        pred_files = sorted(PREDICTIONS_DIR.glob("*.parquet"))
        if not pred_files:
            return {"error": f"No .parquet files found in {PREDICTIONS_DIR}"}

        latest = pred_files[-1]
        print(f"[recsys] usando archivo de predicciones: {latest}")

        # 3) Leer el parquet
        df = pd.read_parquet(latest)
        print(f"[recsys] columnas del parquet: {list(df.columns)}")

        # 4) Verificar columnas necesarias
        required_cols = {"customer_id", "product_id", "proba_compra"}
        missing = required_cols.difference(df.columns)
        if missing:
            return {
                "error": (
                    f"Faltan columnas en {latest.name}: {missing}. "
                    f"Columnas disponibles: {list(df.columns)}"
                )
            }

        # 5) Filtrar por cliente
        df_client = df[df["customer_id"] == req.customer_id]
        if df_client.empty:
            return {
                "error": f"Cliente {req.customer_id} no encontrado en {latest.name}"
            }

        # 6) Top 5 productos con mayor probabilidad
        df_top5 = (
            df_client.sort_values("proba_compra", ascending=False)
            .head(5)[["product_id", "proba_compra"]]
        )

        # Convertimos a tipos nativos (int/float) para que sean JSON-serializables
        recomendaciones = [
            {
                "product_id": int(row["product_id"]),
                "proba_compra": float(row["proba_compra"]),
            }
            for _, row in df_top5.iterrows()
        ]

        return {
            "cliente": int(req.customer_id),
            "archivo": latest.name,
            "recomendaciones": recomendaciones,
        }

    except Exception as e:
        # Si igual algo explota, devolvemos el mensaje en vez de 500 “ciego”
        import traceback
        traceback.print_exc()
        return {"error": f"Exception in /recommend: {e}"}
