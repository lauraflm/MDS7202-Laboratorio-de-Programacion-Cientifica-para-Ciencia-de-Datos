import os
import json
import requests
import gradio as gr

# URL del backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def predict_customer_product(customer_id, product_id, semana, size, num_deliver, x_coord, y_coord, 
                           compro_anterior, promedio_compra, customer_type, sub_category, 
                           segment, package, brand):
    """
    Función para hacer predicciones usando el endpoint estructurado
    """
    try:
        payload = {
            "customer_id": int(customer_id),
            "product_id": int(product_id),
            "semana": int(semana),
            "size": float(size) if size else None,
            "num_deliver_per_week": float(num_deliver) if num_deliver else None,
            "X": float(x_coord) if x_coord else None,
            "Y": float(y_coord) if y_coord else None,
            "compro_semana_pasada": float(compro_anterior),
            "promedio_compra": float(promedio_compra),
            "customer_type": customer_type,
            "sub_category": sub_category,
            "segment": segment,
            "package": package,
            "brand": brand
        }
        
        print(f"DEBUG Frontend: Enviando payload: {payload}")  # Para debug
        
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        print(f"DEBUG Frontend: Status code: {resp.status_code}")  # Para debug
        resp.raise_for_status()
        result = resp.json()
        print(f"DEBUG Frontend: Respuesta: {result}")  # Para debug
        
        return f"""
**Predicción de Compra SodAI Drinks**

**Cliente:** {customer_id} | **Producto:** {product_id}

**Resultados:**
- **Probabilidad de compra:** {result['prediction_proba']:.3f} ({result['prediction_proba']*100:.1f}%)
- **Predicción binaria:** {'SI comprará' if result['prediction_binary'] == 1 else 'NO comprará'}

**Interpretación:**
- {'Alta probabilidad' if result['prediction_proba'] > 0.7 else 'Probabilidad media' if result['prediction_proba'] > 0.3 else 'Baja probabilidad'} de que el cliente compre este producto.
- Recomendación: {'Incluir en campaña de marketing' if result['prediction_proba'] > 0.5 else 'No priorizar para esta campaña'}
        """
        
    except requests.RequestException as e:
        return f"Error de conexión: {e}"
    except Exception as e:
        return f"Error: {e}"


def predict_json(features_json: str) -> str:
    """
    Función para compatibilidad con entrada JSON manual
    """
    try:
        data = json.loads(features_json)
        if not isinstance(data, dict):
            return "❌ El JSON debe ser un objeto con los features como claves."

        payload = {"features": data}
        resp = requests.post(f"{BACKEND_URL}/predict_simple", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        return f"""
**Predicción con JSON**

**Resultado:** {result['prediction']:.3f} ({result['prediction']*100:.1f}%)
**Binario:** {'SI' if result['prediction_binary'] == 1 else 'NO'}

**JSON completo:**
{json.dumps(result, indent=2, ensure_ascii=False)}
        """
        
    except json.JSONDecodeError:
        return "Error: el texto ingresado no es un JSON válido."
    except requests.RequestException as e:
        return f"Error al comunicarse con el backend: {e}"
    except Exception as e:
        return f"Error procesando la respuesta: {e}"


# Configuración de la interfaz
with gr.Blocks(title="SodAI Drinks - Predictor de Compras", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # SodAI Drinks - Predictor de Compras
    
    **Bienvenido al sistema de predicción de compras de SodAI Drinks!**
    
    Esta aplicación utiliza machine learning para predecir la probabilidad de que un cliente específico 
    compre un producto determinado en la próxima semana.
    """)
    
    with gr.Tabs():
        # Tab 1: Predicción estructurada (fácil de usar)
        with gr.TabItem("Predicción por Cliente-Producto"):
            gr.Markdown("### Ingresa los datos del cliente y producto para obtener una predicción")
            
            with gr.Row():
                with gr.Column():
                    customer_id = gr.Number(label="ID Cliente", value=1, precision=0)
                    product_id = gr.Number(label="ID Producto", value=1, precision=0)
                    semana = gr.Number(label="Semana", value=1, precision=0, minimum=1, maximum=52)
                    
                with gr.Column():
                    size = gr.Number(label="Tamaño del producto (L)", value=0.33, step=0.01)
                    num_deliver = gr.Number(label="Entregas por semana", value=3, precision=0, minimum=2, maximum=5)
                    x_coord = gr.Number(label="Coordenada X", value=-107.90, step=0.01)
                    y_coord = gr.Number(label="Coordenada Y", value=-46.56, step=0.01)
            
            with gr.Row():
                with gr.Column():
                    compro_anterior = gr.Number(label="Compró semana anterior (0 o 1)", value=0, minimum=0, maximum=1)
                    promedio_compra = gr.Number(label="Promedio histórico de compra", value=0.02, minimum=0, maximum=1)
                    
                with gr.Column():
                    customer_type = gr.Dropdown(
                        choices=["ABARROTES", "CANAL FRIO", "MAYORISTA", "MINIMARKET", "RESTAURANT", "SUPERMERCADO", "TIENDA DE CONVENIENCIA"],
                        label="Tipo de cliente",
                        value="MINIMARKET"
                    )
                    sub_category = gr.Dropdown(
                        choices=["AGUAS SABORIZADAS", "GASEOSAS"],
                        label="Subcategoría",
                        value="GASEOSAS"
                    )
                    
            with gr.Row():
                segment = gr.Dropdown(
                    choices=["HIGH", "LOW", "MEDIUM", "PREMIUM"],
                    label="Segmento",
                    value="MEDIUM"
                )
                package = gr.Dropdown(
                    choices=["BOTELLA", "KEG", "LATA"],
                    label="Tipo de paquete",
                    value="BOTELLA"
                )
                brand = gr.Dropdown(
                    choices=["Brand 1", "Brand 2", "Brand 3", "Brand 7", "Brand 10", "Brand 14", "Brand 15", "Brand 16", "Brand 21", "Brand 22", "Brand 24", "Brand 26", "Brand 28", "Brand 31", "Brand 33", "Brand 34", "Brand 35", "Brand 44", "Brand 52", "Brand 54", "Brand 55"],
                    label="Marca",
                    value="Brand 1"
                )
            
            predict_btn = gr.Button("Predecir Compra", variant="primary")
            prediction_output = gr.Markdown()
            
            predict_btn.click(
                fn=predict_customer_product,
                inputs=[customer_id, product_id, semana, size, num_deliver, x_coord, y_coord,
                       compro_anterior, promedio_compra, customer_type, sub_category, 
                       segment, package, brand],
                outputs=prediction_output
            )
        
        # Tab 2: Entrada JSON (para usuarios avanzados)
        with gr.TabItem("Entrada JSON Avanzada"):
            gr.Markdown("""
            ### Para usuarios avanzados: Entrada JSON
            
            **Formato esperado:**
            ```json
            {
                "customer_id": 123,
                "product_id": 456,
                "semana": 1,
                "size": 0.33,
                "compro_semana_pasada": 1.0,
                "promedio_compra": 0.3
            }
            ```
            """)
            
            with gr.Row():
                with gr.Column():
                    json_input = gr.Textbox(
                        label="JSON de features",
                        lines=15,
                        value='{\n  "customer_id": 1,\n  "product_id": 1,\n  "semana": 1,\n  "size": 0.33,\n  "compro_semana_pasada": 0.0,\n  "promedio_compra": 0.02\n}',
                        placeholder="Escribe aquí tu JSON con los features..."
                    )
                    json_predict_btn = gr.Button("Predecir con JSON", variant="secondary")
                    
                with gr.Column():
                    json_output = gr.Markdown()
            
            json_predict_btn.click(
                fn=predict_json,
                inputs=json_input,
                outputs=json_output
            )
    
    gr.Markdown("""
    ---
    
    ### Información sobre las variables:
    
    **Variables de Identificación:**
    - **ID Cliente/Producto:** Identificadores únicos en el sistema SodAI
    - **Semana:** Semana del año (1-52) para la predicción
    
    **Variables Numéricas:**
    - **Tamaño:** Tamaño del producto en litros (rango: 0.25L - 20L)
    - **Entregas por semana:** Frecuencia de entregas (2-5 veces por semana)
    - **Coordenadas X/Y:** Ubicación geográfica del cliente
    - **Compró semana anterior:** Si el cliente compró (1) o no (0) la semana anterior
    - **Promedio histórico:** Promedio de compras del cliente (0.0 a 1.0)
    
    **Variables Categóricas:**
    - **Tipo de cliente:** ABARROTES, CANAL FRIO, MAYORISTA, MINIMARKET, RESTAURANT, SUPERMERCADO, TIENDA DE CONVENIENCIA
    - **Subcategoría:** AGUAS SABORIZADAS, GASEOSAS
    - **Segmento:** HIGH, LOW, MEDIUM, PREMIUM
    - **Paquete:** BOTELLA, KEG, LATA
    - **Marca:** Brand 1 hasta Brand 55 (21 marcas disponibles)
    
    **Tip:** Todos los valores están basados en el dataset real de SodAI Drinks.
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)