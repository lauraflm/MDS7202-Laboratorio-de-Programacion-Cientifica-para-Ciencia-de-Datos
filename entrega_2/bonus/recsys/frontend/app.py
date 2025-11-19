import gradio as gr
import requests

BACKEND_URL = "http://recsys-backend:8001/recommend"

def recomendar(cliente_id):
    payload = {"customer_id": int(cliente_id)}
    resp = requests.post(BACKEND_URL, json=payload)
    if resp.status_code != 200:
        return "Error en backend"
    data = resp.json()
    if "error" in data:
        return data["error"]
    recs = data["recomendaciones"]
    texto = "\n".join([f"- Producto {r['product_id']}: {r['proba_compra']:.3f}" for r in recs])
    return texto

demo = gr.Interface(
    fn=recomendar,
    inputs=gr.Number(label="ID del cliente"),
    outputs=gr.Textbox(label="Top 5 recomendaciones"),
    title="Sistema de Recomendación SodAI",
)

demo.launch(server_name="0.0.0.0", server_port=7861)
