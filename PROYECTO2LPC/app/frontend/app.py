import os
import json
import requests
import gradio as gr

# url backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")


def call_backend(features_json: str) -> str:
    try:
        data = json.loads(features_json)
        if not isinstance(data, dict):
            return "El JSON debe ser un objeto con los features como claves."

        payload = {"features": data}

        resp = requests.post(BACKEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        return json.dumps(result, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return "Error: el texto ingresado no es un JSON válido."
    except requests.RequestException as e:
        return f"Error al comunicarse con el backend: {e}"
    except Exception as e:
        return f"Ocurrió un error procesando la respuesta: {e}"


description_text = """
Bienvenido a SodAI Drinks.

Esta interfaz permite interactuar con el modelo desplegado a través del backend FastAPI.

Instrucciones de uso:
1. En el cuadro de texto de entrada, escribe un JSON con los features que requiere el modelo.
   Ejemplo:
   {
     "feature_1": 1.0,
     "feature_2": 2.5
   }

2. Presiona el botón "Obtener predicción".
3. El resultado crudo de la API aparecerá en el cuadro de salida, mostrando la predicción.
"""

with gr.Blocks(title="SodAI Drinks Frontend") as demo:
    gr.Markdown("# SodAI Drinks")
    gr.Markdown(description_text)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="JSON de features de entrada",
                lines=10,
                value='{\n  "feature_1": 1.0,\n  "feature_2": 2.5\n}',
            )
            submit_btn = gr.Button("Obtener predicción")
        with gr.Column():
            output_box = gr.Textbox(
                label="Respuesta del backend",
                lines=10,
            )

    submit_btn.click(
        fn=call_backend,
        inputs=input_box,
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
