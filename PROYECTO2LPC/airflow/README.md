# Pipeline Productivo – SodAI Drinks (Airflow)

Este documento describe el pipeline productivo que se hizo en Apache Airflow como parte de la Entrega 2, para así poder agilizar el proceso de procesamiento, entrenamiento y predicción de los datos. 
El DAG incluye todo el flujo desde la preparación de datos hasta la generación de predicciones semanales, incluyendo detección de drift y reentrenamiento automático del modelo cuando es necesario hacerlo.



# 1. Descripción del DAG

El DAG implementa un flujo end-to-end que se ejecuta semanalmente (`@weekly`) y cumple con:

- Se introducen automáticamente los nuevos datos.
- Generación de features y reconstrucción del dataset semanal.
- Detección de drift entre la data utilizada para las semanas existentes y la nueva data posible.
- Reentrenamiento automático del modelo cuando se deba hacer.
- Generación de predicciones para la **semana siguiente a la última disponible (t+1)**.
- Persistencia de modelos, métricas y predicciones.

El DAG se llama sodai_pipeline_dag, debido a que es para la empresa sodai.


y su estructura es:

- `start`
- `build_weekly_features`
- `branch_drift`
    - `train_model` (si hay drift)
    - `skip_train` (si no hay drift)
- `join_after_train`
- `generate_predictions`
- `end`

---

# 2. Explicación Detallada de Cada Tarea

### **1. start**
Operador vacío que marca el inicio del pipeline.  
Se usa para mantener una estructura clara y estandarizada.

---

### **2. build_weekly_features**  
`PythonOperator`

En esta parte se ejetuta la función `build_weekly_features()` ubicada en `src/preprocessing.py`. La cual crea las semanas y las features correspondientes a ellas.

Esta etapa:

- Lee automáticamente todos los archivos `transacciones*.parquet` en `data/raw/`, incluyendo semanas nuevas.
- Construye la base **cliente × producto × semana**.
- Genera variables:
  - `compro_semana_pasada`
  - `promedio_compra`
- Realiza imputación, clipping, sin/cos para semana y encoding categórico.
- Crea el split temporal:  
  **train (1–36), val (37–44), test (45–53)**.
- Guarda `weekly_features.parquet` para las siguientes etapas.

Esta etapa asegura que el pipeline pueda integrar nuevos datos en la semana t+1 sin modificar código.

---

### **3. branch_drift**  
`BranchPythonOperator`

Ejecuta `detect_drift()` desde `src/drift.py`, el cual:

- Calcula PSI/IQR u otro método sobre las columnas numéricas.
- Compara la distribución de datos nuevos vs históricos.
- Retorna `True` si hay drift significativo.

Dependiendo del resultado, Airflow decide:

- `train_model` si es que hay drift  
- `skip_train` si es que no hay drift

Esto crea un flujo condicional y productivo dependiendo de los datos con los cuales se encuentre trabajando y el objetivo que se busque, simulando un sistema real.

---

### **4. train_model**
`PythonOperator`

Ejecuta `train_model()` desde `src/training.py`.

Esta etapa:

- Carga las features procesadas.
- Genera pipelines **preprocessing + XGBoost**.
- Ajusta hiperparámetros razonables (SMW incluido).
- Entrena el modelo con los datos más recientes.
- Registra métricas:
  - AUC
  - Average Precision
  - Precision/Recall
  - F1
- Guarda el modelo en:  
  `airflow/artifacts/models/xgb_pipeline.joblib`.
- Guarda métricas en:  
  `airflow/artifacts/metrics_train.json`.

Si no se ejecuta (porque no había drift), la tarea `skip_train` permite continuar el DAG.

---

### **5. skip_train**
`EmptyOperator`

Se ejecuta solo si no hay drift. Con este paso, se logra evitar reentrenamientos innecesarios.

---

### **6. join_after_train**
`EmptyOperator` con `TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS`.

Permite sincronizar ambos flujos (con y sin reentrenamiento) para continuar el pipeline.

---

### **7. generate_predictions**
`PythonOperator`

Ejecuta `generate_predictions()` desde `src/predict.py`.

Esta etapa:

1. Carga `weekly_features.parquet`.
2. Determina la última semana (t) disponible.
3. Construye un dataset para la semana t+1.
4. Usa el modelo más actualizado (entrenado o previo).
5. Genera probabilidades `proba_compra` para cada cliente–producto.
6. Guarda un archivo único por semana:

airflow/artifacts/predictions/predicciones_semana_{t+1}.parquet


Esto completa el flujo **t → t+1 → t+2**, como se busca.

---

### **8. end**
Operador vacío que marca el fin del pipeline.

---

# 3. Diagrama de Flujo

```mermaid
flowchart TD
    A[start] --> B[build_weekly_features]

    B --> C{branch_drift}

    C -->|drift| D[train_model]
    C -->|no drift| E[skip_train]

    D --> F[join_after_train]
    E --> F[join_after_train]

    F --> G[generate_predictions] --> H[end]

4. Representación visual del DAG

La arquitectura del pipeline fue diseñada para simular un entorno productivo real, donde semanalmente llegan nuevos datos y el sistema debe decidir si reentrenar o no, manteniendo la estabilidad del modelo en el tiempo. A continuación se detalla cómo se implementaron estos tres componentes clave.

**Integración Automática de Nuevos Datos**

El pipeline está preparado para incorporar nuevas semanas de datos sin intervención manual.
Esto se logra gracias a la función load_raw_data() ubicada en src/data_io.py, la cual carga todos los archivos de transacciones que existan en la carpeta data/raw/:

tx_files = sorted(RAW_DIR.glob("transacciones*.parquet"))
dfs = [pd.read_parquet(f) for f in tx_files]
transacciones = pd.concat(dfs, ignore_index=True)


Esto significa que, si aparece un archivo nuevo como:
- transacciones_2024_15.parquet
- transacciones_nuevos_datos.parquet
el DAG lo incorporará automáticamente en la próxima ejecución, reconstruirá la base semanal utilizando todos los datos históricos + los nuevos y no requiere modificar código.


**Detección de Drift entre Datos Antiguos y Nuevos**

El módulo src/drift.py implementa un mecanismo para detectar data drift.
A partir de métricas como PSI (Population Stability Index), se comparan:
la distribución del dataset antiguo (train), con la distribución del dataset nuevo (semanas recientes).

Ejemplo simplificado:

psis[col] = _psi(df_train[col], df_test[col])
psi_mean = np.mean(list(psis.values()))
has_drift = psi_mean > threshold

Si psi_mean supera el umbral configurado, se considera que existe drift.
Este resultado (True/False) es devuelto al DAG mediante el BranchPythonOperator.

**Reentrenamiento Automático Condicional**

El DAG utiliza un BranchPythonOperator para decidir si reentrenar:
return "train_model" if has_drift else "skip_train"

* Si hay drift:
Se ejecuta train_model, donde el pipeline reentrena el modelo, recalcula métricas, guarda una nueva versión del modelo productivo.

* Si no hay drift:
Se ejecuta skip_train, evitando un reentrenamiento innecesario. Ambas rutas confluyen en join_after_train gracias a: 
trigger_rule = TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS


**Generación de Predicciones para la Semana Siguiente (t+1)**

La etapa final generate_predictions realiza:
- Detectar la última semana presente en los datos (t).
- Construir una base de scoring para la semana futura (t+1).
- Aplicar el modelo más reciente (entrenado o previo).

Guardar las predicciones en: artifacts/predictions/predicciones_semana_{t+1}.parquet
