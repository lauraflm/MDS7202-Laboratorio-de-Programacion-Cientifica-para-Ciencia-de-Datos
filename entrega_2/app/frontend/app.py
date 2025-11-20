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
- **Recomendación:** {'Incluir en campaña de marketing' if result['prediction_proba'] > 0.5 else 'No priorizar para esta campaña'}
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
            return "Error: El JSON debe ser un objeto con los features como claves."

        payload = {"features": data}
        resp = requests.post(f"{BACKEND_URL}/predict_simple", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        return f"""
**Predicción con JSON**

**Resultado:** {result['prediction']:.3f} ({result['prediction']*100:.1f}%)
**Binario:** {'SI' if result['prediction_binary'] == 1 else 'NO'}

**JSON completo:**
```json
{json.dumps(result, indent=2, ensure_ascii=False)}
```
        """
        
    except json.JSONDecodeError:
        return "Error: el texto ingresado no es un JSON válido."
    except requests.RequestException as e:
        return f"Error al comunicarse con el backend: {e}"
    except Exception as e:
        return f"Error procesando la respuesta: {e}"


def predict_next_week_automatic(threshold: float, semana: int) -> str:
    """
    Función para predecir automáticamente todas las duplas cliente-producto 
    que comprarán en la semana especificada
    """
    try:
        # Llamar al endpoint de predicción automática
        params = {"threshold": threshold, "semana": semana}
        resp = requests.post(f"{BACKEND_URL}/predict_next_week", params=params, timeout=30)
        print(f"DEBUG Frontend: Status code: {resp.status_code}")
        resp.raise_for_status()
        result = resp.json()
        print(f"DEBUG Frontend: Respuesta recibida con {result['total_duplas']} duplas")
        
        # Formatear la respuesta como tabla HTML
        output = f"""
**Predicción Automática - Semana {semana}**

**Semana predicha:** {result['semana_predicha']}  
**Umbral usado:** {result['umbral_usado']}  
**Total de duplas que comprarán:** {result['total_duplas']}

---

### Duplas Cliente-Producto Recomendadas

<div style="max-height: 400px; overflow-y: auto; border: 2px solid #0066CC; border-radius: 10px; padding: 15px; background-color: #2d2d2d;">
<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
<thead>
<tr style="background-color: #0066CC; color: white;">
<th style="padding: 12px; border: 1px solid #0066CC; text-align: center; border-radius: 5px;">Ranking</th>
<th style="padding: 12px; border: 1px solid #0066CC; text-align: center; border-radius: 5px;">Cliente ID</th>
<th style="padding: 12px; border: 1px solid #0066CC; text-align: center; border-radius: 5px;">Producto ID</th>
<th style="padding: 12px; border: 1px solid #0066CC; text-align: center; border-radius: 5px;">Probabilidad</th>
<th style="padding: 12px; border: 1px solid #0066CC; text-align: center; border-radius: 5px;">Recomendación</th>
</tr>
</thead>
<tbody>
"""
        
        # Agregar filas de la tabla
        for i, dupla in enumerate(result['duplas_predichas'], 1):
            proba_percent = dupla['prediction_proba'] * 100
            
            # Color de la fila basado en la probabilidad - TONOS AZULES
            if proba_percent >= 80:
                row_color = "#1a4480"  # Azul más intenso
                recomendacion = "Prioritario"
            elif proba_percent >= 60:
                row_color = "#2d5aa0"  # Azul medio  
                recomendacion = "Recomendado"
            else:
                row_color = "#4070c0"  # Azul claro
                recomendacion = "Considerar"
            
            output += f"""
<tr style="background-color: {row_color}; color: white;">
<td style="padding: 10px; border: 1px solid #0066CC; text-align: center; font-weight: bold;">#{i}</td>
<td style="padding: 10px; border: 1px solid #0066CC; text-align: center;">{dupla['customer_id']}</td>
<td style="padding: 10px; border: 1px solid #0066CC; text-align: center;">{dupla['product_id']}</td>
<td style="padding: 10px; border: 1px solid #0066CC; text-align: center; font-weight: bold;">{proba_percent:.1f}%</td>
<td style="padding: 10px; border: 1px solid #0066CC; text-align: center;">{recomendacion}</td>
</tr>
"""
        
        output += """
</tbody>
</table>
</div>

---

### Interpretación de Resultados

- **Prioritario (≥80%):** Duplas con muy alta probabilidad - enfocar campañas aquí
- **Recomendado (60-79%):** Duplas con buena probabilidad - incluir en estrategia  
- **Considerar (≤59%):** Duplas con probabilidad moderada - evaluar según recursos

**Acción recomendada:** Enfocar esfuerzos comerciales en estos {result['total_duplas']} pares cliente-producto para maximizar conversión en la próxima semana.
        """
        
        return output
        
    except requests.RequestException as e:
        return f"**Error de conexión con el backend:** {e}"
    except Exception as e:
        return f"**Error procesando la predicción automática:** {e}"


# Configuración de la interfaz con tema personalizado
with gr.Blocks(
    title="SodAI Drinks - Predictor de Compras", 
    theme="soft",
    css="""
    /* MODO OSCURO - AZUL ELEGANTE Y CONSISTENTE */
    
    /* Fondo general - Oscuro */
    .gradio-container {
        background: #1a1a1a !important;
        min-height: 100vh;
        font-family: Arial, sans-serif !important;
        color: #e0e0e0 !important;
    }
    
    /* HEADER - Cambiar fondo rosa/blanco por oscuro */
    .gradio-container .gr-blocks .gr-row:first-child,
    .gradio-container header,
    .gr-interface header {
        background: #2d2d2d !important;
        border: 1px solid #0066CC !important;
        border-radius: 8px !important;
        margin-bottom: 20px !important;
    }
    
    /* Contenido principal - Oscuro */
    .gr-blocks {
        background: #2d2d2d !important;
        border-radius: 10px !important;
        margin: 20px !important;
        padding: 30px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        color: #e0e0e0 !important;
    }
    
    /* TODOS LOS BOTONES - AZUL ELEGANTE */
    .gr-button-primary, .gr-button, button {
        background: #0066CC !important;
        border: 1px solid #0066CC !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3) !important;
    }
    
    /* Hover REDUCIDO en todos los botones */
    .gr-button-primary:hover, .gr-button:hover, button:hover {
        background: #0052A3 !important;
        box-shadow: 0 3px 10px rgba(0, 102, 204, 0.4) !important;
        transform: translateY(-1px) !important;
        border: 1px solid #0052A3 !important;
    }
    
    /* Títulos - Azul elegante */
    h1, h2, h3, h4 {
        color: #0066CC !important;
        font-weight: bold !important;
    }
    
    /* Pestañas - Solo azul y gris */
    .gr-tabs {
        border: 1px solid #4d4d4d !important;
        border-radius: 8px !important;
        background: #2d2d2d !important;
    }
    
    /* Pestaña individual - Gris */
    .gr-tab, .gr-tab-nav button {
        background: #3d3d3d !important;
        color: #b0b0b0 !important;
        border: 1px solid #4d4d4d !important;
        font-weight: normal !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 15px !important;
    }
    
    /* Pestaña seleccionada - AZUL ELEGANTE */
    .gr-tab.selected, .gr-tab-nav button[aria-selected="true"] {
        background: #0066CC !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid #0066CC !important;
    }
    
    /* COMPONENTES SVELTE/GRADIO ESPECÍFICOS */
    
    /* Sliders nativos - Azul elegante */
    input[type="range"] {
        accent-color: #1a4480 !important;
        background: #3d3d3d !important;
        border-radius: 6px !important;
    }
    
    /* Number inputs nativos */
    input[type="number"] {
        background: #3d3d3d !important;
        border: 1px solid #4d4d4d !important;
        color: #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    input[type="number"]:focus {
        border-color: #0066CC !important;
        outline: none !important;
        box-shadow: 0 0 5px rgba(0, 102, 204, 0.4) !important;
    }
    
    /* Select/Dropdown nativos */
    select {
        background: #3d3d3d !important;
        border: 1px solid #4d4d4d !important;
        color: #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    select:focus {
        border-color: #0066CC !important;
        outline: none !important;
    }
    
    /* Labels de componentes Svelte */
    .gr-form label, .gr-box label {
        color: #0066CC !important;
        font-weight: normal !important;
    }
    
    /* Paneles - Oscuro */
    .gr-panel {
        border: 1px solid #4d4d4d !important;
        border-radius: 6px !important;
        background: #333333 !important;
        padding: 15px !important;
    }
    
    /* Inputs generales */
    .gr-textbox, .gr-number, .gr-dropdown {
        border: 1px solid #4d4d4d !important;
        border-radius: 4px !important;
        background: #3d3d3d !important;
        color: #e0e0e0 !important;
    }
    
    .gr-textbox:focus, .gr-number:focus, .gr-dropdown:focus {
        border-color: #0066CC !important;
        box-shadow: 0 0 5px rgba(0, 102, 204, 0.4) !important;
        background: #404040 !important;
    }
    
    /* Slider containers */
    .gr-slider, input[type="range"] {
        accent-color: #0066CC !important;
        background: #3d3d3d !important;
    }
    
    .gr-slider-container, .gr-slider .gr-slider-track, .gr-form {
        background: #3d3d3d !important;
        border: 1px solid #4d4d4d !important;
        border-radius: 6px !important;
    }
    
    /* Labels del slider - AZUL ELEGANTE */
    .gr-slider label, .gr-slider span {
        color: #0066CC !important;
        font-weight: bold !important;
    }
    
    /* Etiquetas - Gris claro */
    .gr-label, label {
        color: #b0b0b0 !important;
        font-weight: normal !important;
    }
    
    /* Texto normal - Gris claro */
    .gr-markdown p, p, span, div {
        color: #e0e0e0 !important;
        font-weight: normal !important;
    }
    
    /* Texto destacado - AZUL ELEGANTE */
    .gr-markdown strong, .gr-markdown b, strong, b {
        color: #0066CC !important;
        font-weight: bold !important;
    }
    
    /* Contenedor de output - Oscuro */
    .gr-markdown {
        background: #333333 !important;
        border: 1px solid #4d4d4d !important;
        border-radius: 6px !important;
        padding: 15px !important;
        color: #e0e0e0 !important;
    }
    
    /* Acordeones - AZUL ELEGANTE */
    .gr-accordion {
        border: 1px solid #0066CC !important;
        border-radius: 6px !important;
        background: #2d2d2d !important;
        margin: 10px 0 !important;
    }
    
    /* Header del acordeón - AZUL ELEGANTE */
    .gr-accordion summary {
        background: #0066CC !important;
        color: white !important;
        font-weight: bold !important;
        padding: 12px 15px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
    }
    
    /* Contenido del acordeón - Oscuro */
    .gr-accordion details[open] > div {
        background: #333333 !important;
        padding: 15px !important;
        color: #e0e0e0 !important;
    }
    
    /* Dropdown options - Oscuro */
    .gr-dropdown select {
        background: #3d3d3d !important;
        color: #e0e0e0 !important;
    }
    
    /* Números en inputs - Claro */
    .gr-number input {
        color: #e0e0e0 !important;
        background: #3d3d3d !important;
    }
    
    /* Scrollbars y elementos varios */
    * {
        scrollbar-color: #0066CC #3d3d3d !important;
    }
    
    /* Forzar azul elegante en elementos que puedan tener otros colores */
    .gr-button-primary, button[variant="primary"] {
        background: #0066CC !important;
        border-color: #0066CC !important;
    }
    
    /* Info y elementos de ayuda - AZUL ELEGANTE */
    .gr-info, .gr-textbox .gr-textbox-info {
        color: #0066CC !important;
    }
    
    /* Links y elementos interactivos - AZUL ELEGANTE */
    a, .gr-link {
        color: #0066CC !important;
    }
    
    /* Forzar que no haya elementos con colores incorrectos */
    .gr-tab-nav button, .gr-button, .gr-slider, input {
        color: inherit !important;
    }
    """
) as demo:
    gr.Markdown("""
    # SodAI Drinks - Predictor de Compras
    
    **Bienvenido al sistema de predicción de compras de SodAI Drinks**
    
    Esta aplicación utiliza machine learning para predecir qué clientes comprarán qué productos en la próxima semana.
    """)
    
    with gr.Tabs():
        # Tab 1: Predicción Automática (PRINCIPAL)
        with gr.TabItem("Predicción Automática de Próxima Semana"):
            gr.Markdown("""
            ## Predicción Automática para la Próxima Semana
            
            Esta función analiza **automáticamente todas las combinaciones cliente-producto** 
            y predice cuáles realizarán una compra en la próxima semana disponible.
            """)
            
            with gr.Accordion("¿Cómo funciona?", open=False):
                gr.Markdown("""
                1. El sistema crea todas las combinaciones posibles cliente-producto
                2. Aplica el modelo entrenado a cada combinación  
                3. Filtra solo las duplas con probabilidad superior al umbral
                4. Retorna la lista ordenada por probabilidad
                
                **Casos de uso:**
                - Campañas de marketing dirigido
                - Planificación de inventario
                - Estrategias de ventas semanales
                """)
            
            with gr.Row():
                with gr.Column():
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        label="Umbral de probabilidad",
                        info="Solo mostrar duplas con probabilidad >= umbral"
                    )
                    
                    semana_selector = gr.Number(
                        label="Semana a predecir",
                        value=54,
                        precision=0,
                        minimum=1,
                        maximum=60,
                        info="Ingresa la semana específica para la predicción"
                    )
                    
                    auto_predict_btn = gr.Button(
                        "Generar Predicciones Automáticas", 
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    **Nota:** Esta operación puede tomar unos segundos ya que 
                    analiza todas las combinaciones posibles.
                    """)
                    
                with gr.Column():
                    auto_prediction_output = gr.Markdown()
            
            auto_predict_btn.click(
                fn=predict_next_week_automatic,
                inputs=[threshold_slider, semana_selector],
                outputs=auto_prediction_output
            )
        
        # Tab 2: Predicción Individual
        with gr.TabItem("Predicción por Cliente-Producto Individual"):
            gr.Markdown("### Ingresa los datos del cliente y producto para obtener una predicción específica")
            
            # Variables básicas siempre visibles
            with gr.Row():
                customer_id = gr.Number(label="ID Cliente", value=1, precision=0)
                product_id = gr.Number(label="ID Producto", value=1, precision=0)
                semana = gr.Number(label="Semana", value=1, precision=0, minimum=1, maximum=52)
            
            # Variables avanzadas en desplegable
            with gr.Accordion("Variables Avanzadas", open=False):
                with gr.Row():
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
        
        # Tab 3: Entrada JSON
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
    
    **Tip:** Todos los valores están basados en el dataset real de SodAI Drinks.""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)