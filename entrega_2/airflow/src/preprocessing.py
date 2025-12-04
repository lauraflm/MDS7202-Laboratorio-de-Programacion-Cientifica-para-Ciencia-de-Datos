# src/preprocessing.py

import numpy as np
import pandas as pd

from .data_io import load_raw_data, save_features

# =====================================================
# 1. Transformadores auxiliares
# =====================================================

class IQRClipper:
    """
    Recorta outliers numéricos usando rango intercuartílico (IQR).
    Compatible con sklearn Pipeline.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.bounds_ = []
        for c in X.columns:
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo = q1 - self.factor * iqr
            hi = q3 + self.factor * iqr
            self.bounds_.append((lo, hi))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i, c in enumerate(X.columns):
            lo, hi = self.bounds_[i]
            X[c] = X[c].clip(lo, hi)
        return X.values


class ComportamientoCompraTransformer:
    """
    Agrega variables de comportamiento por cliente:

    - compro_semana_pasada: 1 si el cliente compró la semana anterior.
    - promedio_compra: promedio histórico del target hasta la semana anterior.

    Compatible con sklearn Pipeline.
    """

    def __init__(
        self,
        customer_col: str = "customer_id",
        semana_col: str = "semana",
        target_col: str = "compra_flag",
    ):
        self.customer_col = customer_col
        self.semana_col = semana_col
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_transformed = X.copy()

        # -------- 1) compro_semana_pasada --------
        compras_por_semana = (
            X_transformed[X_transformed[self.target_col] == 1]
            .groupby([self.customer_col, self.semana_col])
            .size()
            .reset_index(name="compro_flag_temp")
        )
        compras_por_semana["semana_anterior"] = compras_por_semana[self.semana_col] - 1

        compro_anterior = compras_por_semana[
            [self.customer_col, "semana_anterior"]
        ].rename(columns={"semana_anterior": self.semana_col})
        compro_anterior["compro_semana_pasada"] = 1

        X_transformed = X_transformed.merge(
            compro_anterior[
                [self.customer_col, self.semana_col, "compro_semana_pasada"]
            ],
            on=[self.customer_col, self.semana_col],
            how="left",
        )
        X_transformed["compro_semana_pasada"] = (
            X_transformed["compro_semana_pasada"].fillna(0).astype(int)
        )

        # -------- 2) promedio_compra (acumulado hasta semana anterior) --------
        X_sorted = X_transformed.sort_values([self.customer_col, self.semana_col])

        expanding_mean = (
            X_sorted.groupby(self.customer_col)[self.target_col]
            .expanding()
            .mean()
            .shift(1)  # no incluye la semana actual
            .fillna(0)
            .values
        )

        X_sorted["promedio_compra"] = expanding_mean
        X_transformed = X_sorted.sort_index()

        return X_transformed


def week_to_sin_cos(a):
    """
    Codificación cíclica para la columna 'semana' (1..52/53).
    Devuelve dos columnas: sin y cos.
    """
    a = np.asarray(a).reshape(-1, 1).astype(float)
    P = 52.0
    sin_ = np.sin(2 * np.pi * (a / P))
    cos_ = np.cos(2 * np.pi * (a / P))
    return np.hstack([sin_, cos_])


def make_ohe(min_freq=None):
    """
    Wrapper para OneHotEncoder compatible con distintas versiones de sklearn.
    """
    from sklearn.preprocessing import OneHotEncoder  # import local

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            **({"min_frequency": min_freq} if min_freq is not None else {}),
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            **({"min_frequency": min_freq} if min_freq is not None else {}),
        )


# ==========================================================
# 2. Construcción de la base semanal (weekly_full_final)
# ==========================================================

def _build_weekly_full_final(df_clientes, df_productos, df_transacciones):
    """
    Construye weekly_full_final (base semanal cliente × producto × semana):

    - Base C×P×semana (todas las semanas presentes en transacciones).
    - items_cp, pedidos_cp, compra_flag.
    - Features semanales agregadas por cliente y producto.
    - Merge con productos y clientes.
    """

    tx = df_transacciones.copy()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"]).dt.normalize()
    tx["semana"] = tx["purchase_date"].dt.isocalendar().week.astype(int)
    tx["semana"] = tx["semana"].clip(1, 53)

    base_cp_sem = (
        tx.groupby(["customer_id", "product_id", "semana"], as_index=False)
        .agg(
            items_cp=("items", "sum"),
            pedidos_cp=("order_id", "nunique"),
        )
        .assign(compra_flag=lambda d: (d["items_cp"] > 0).astype("int8"))
    )

    weeks_total = np.sort(tx["semana"].unique())
    customers_total = tx["customer_id"].unique()
    products_total = tx["product_id"].unique()

    idx = pd.MultiIndex.from_product(
        [customers_total, products_total, weeks_total],
        names=["customer_id", "product_id", "semana"],
    )
    full = idx.to_frame(index=False)

    weekly_full = (
        full.merge(
            base_cp_sem[
                ["customer_id", "product_id", "semana", "items_cp", "pedidos_cp"]
            ],
            on=["customer_id", "product_id", "semana"],
            how="left",
            validate="many_to_one",
        )
        .fillna({"items_cp": 0, "pedidos_cp": 0})
    )

    weekly_full["compra_flag"] = (weekly_full["items_cp"] > 0).astype("int8")
    weekly_full["pedidos_cp"] = weekly_full["pedidos_cp"].astype("int16")

    # ===== Features sencillas por semana =====
    weekly_full["items_cliente_semana"] = (
        weekly_full.groupby(["customer_id", "semana"])["items_cp"]
        .transform("sum")
        .astype("int32")
    )

    weekly_full["items_producto_semana"] = (
        weekly_full.groupby(["product_id", "semana"])["items_cp"]
        .transform("sum")
        .astype("int32")
    )

    weekly_full["productos_activos_cliente_semana"] = (
        weekly_full.assign(
            compra_flag_temp=(weekly_full["items_cp"] > 0).astype("int8")
        )
        .groupby(["customer_id", "semana"])["compra_flag_temp"]
        .transform("sum")
        .astype("int16")
    )

    # --- Merge con productos ---
    weekly_enriquecido = weekly_full.merge(
        df_productos,
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    weekly_enriquecido["producto_en_catalogo"] = weekly_enriquecido["brand"].notna()

    # --- Limpieza de df_clientes ---
    df_clientes = df_clientes.copy()
    df_clientes["X"] = pd.to_numeric(df_clientes["X"], errors="coerce")
    df_clientes["Y"] = pd.to_numeric(df_clientes["Y"], errors="coerce")

    for c in ["region_id", "zone_id"]:
        if c in df_clientes.columns:
            df_clientes[c] = pd.to_numeric(df_clientes[c], errors="coerce").astype(
                "Int64"
            )

    cols_cli = [
        "customer_id",
        "customer_type",
        "Y",
        "X",
        "num_deliver_per_week",
        "num_visit_per_week",
    ]
    cols_cli = [c for c in cols_cli if c in df_clientes.columns]

    weekly_full_final = weekly_enriquecido.merge(
        df_clientes[cols_cli],
        on="customer_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_cli"),
    )

    return weekly_full_final


# ==================================================
# 3. Features extra + split temporal
# ==================================================

ID_COLS = ["customer_id", "product_id"]
TARGET_COL = "compra_flag"
WEEK_COL = "semana"
SPLIT_COL = "split"


def _add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega:
      - compro_semana_pasada: 1 si el cliente compró en la semana anterior
      - promedio_compra: promedio histórico de compra hasta la semana anterior

    IMPLEMENTACIÓN EFICIENTE:
      1) Trabajamos primero a nivel cliente-semana (muchas menos filas).
      2) Calculamos ahí las features de comportamiento.
      3) Hacemos merge de vuelta a la tabla cliente-producto-semana.
    """

    print("[PREPROCESS] _add_behavior_features: inicio")

    # Nos aseguramos que las columnas clave existan
    required_cols = ["customer_id", WEEK_COL, TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df: {missing}")

    # ===================================================================
    # 1) Agregamos a nivel cliente-semana: ¿compró algo esa semana?
    # ===================================================================
    # compra_semana = 1 si el cliente compró AL MENOS 1 producto esa semana
    df_cw = (
        df.groupby(["customer_id", WEEK_COL])[TARGET_COL]
        .max()  # como el target es 0/1, max indica si compró algo
        .rename("compra_semana")
        .reset_index()
    )

    # Ordenar por cliente y semana
    df_cw = df_cw.sort_values(["customer_id", WEEK_COL])

    # ===================================================================
    # 2) compro_semana_pasada: shift de compra_semana dentro de cada cliente
    # ===================================================================
    df_cw["compro_semana_pasada"] = (
        df_cw.groupby("customer_id")["compra_semana"]
        .shift(1)               # semana anterior
        .fillna(0)
        .astype("int8")
    )

    # ===================================================================
    # 3) promedio_compra: promedio histórico hasta la semana anterior
    # ===================================================================
    # Primero, promedio acumulado por cliente
    # Usamos compounding sobre "compra_semana"
    exp_mean = (
        df_cw.groupby("customer_id")["compra_semana"]
        .expanding()
        .mean()
        .shift(1)   # hasta la semana anterior (no incluye la actual)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype("float32")
    )

    df_cw["promedio_compra"] = exp_mean

    # Ya no necesitamos "compra_semana" como feature final
    df_cw = df_cw.drop(columns=["compra_semana"])

    print(
        "[PREPROCESS] _add_behavior_features: "
        f"tabla cliente-semana con comportamiento: {df_cw.shape}"
    )

    # ===================================================================
    # 4) Merge de vuelta a la base cliente-producto-semana
    # ===================================================================
    df_out = df.merge(
        df_cw,
        on=["customer_id", WEEK_COL],
        how="left",
        validate="many_to_one",  # cada (cliente, semana) tiene 1 fila en df_cw
    )

    # Fill defensivo (por si acaso hubiera NA)
    df_out["compro_semana_pasada"] = (
        df_out["compro_semana_pasada"].fillna(0).astype("int8")
    )
    df_out["promedio_compra"] = df_out["promedio_compra"].fillna(0.0).astype("float32")

    print(
        "[PREPROCESS] _add_behavior_features: fin, "
        f"shape final {df_out.shape}"
    )

    return df_out



def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features extra por cliente–producto–semana:
      - recency
      - customer_buy_rate
      - product_popularity
      - score_pop_rec
    """
    df = df.sort_values(["customer_id", "product_id", WEEK_COL]).copy()

    # recency
    mask_compra = df[TARGET_COL] == 1
    last_purchase_week = (
        df.where(mask_compra)
        .groupby(["customer_id", "product_id"])[WEEK_COL]
        .ffill()
    )
    df["last_purchase_week"] = last_purchase_week
    df["recency"] = df[WEEK_COL] - df["last_purchase_week"]

    max_recency = df["recency"].max(skipna=True)
    if pd.isna(max_recency):
        max_recency = 52

    df["recency"] = df["recency"].fillna(max_recency + 1).clip(lower=0, upper=52)
    df = df.drop(columns=["last_purchase_week"])

    # customer_buy_rate
    df["customer_buy_rate"] = (
        df.groupby("customer_id")[TARGET_COL].transform("mean").fillna(0.0)
    )

    # product_popularity
    df["product_popularity"] = (
        df.groupby("product_id")[TARGET_COL].transform("sum").astype(float)
    )

    # score_pop_rec
    df["score_pop_rec"] = df["product_popularity"] / (1.0 + df["recency"])

    return df


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    DROP_ALWAYS = ["producto_en_catalogo"]
    DROP_SINGLETONS = []

    if "category" in df.columns and df["category"].nunique(dropna=True) <= 1:
        DROP_SINGLETONS.append("category")

    if (
        "num_visit_per_week" in df.columns
        and df["num_visit_per_week"].nunique(dropna=True) <= 1
    ):
        DROP_SINGLETONS.append("num_visit_per_week")

    to_drop = [c for c in DROP_ALWAYS + DROP_SINGLETONS if c in df.columns]
    return df.drop(columns=to_drop)

def _add_temporal_split(df):
    df = df.copy()
    df[WEEK_COL] = df[WEEK_COL].clip(1, 53)

    weeks = sorted(df[WEEK_COL].unique())
    n = len(weeks)

    w_train = set(weeks[: int(0.7 * n)])
    w_val   = set(weeks[int(0.7 * n): int(0.85 * n)])
    w_test  = set(weeks[int(0.85 * n):])

    def asignar_split(sem):
        if sem in w_train: return "train"
        if sem in w_val:   return "val"
        if sem in w_test:  return "test"
        return "out"

    df[SPLIT_COL] = df[WEEK_COL].apply(asignar_split)
    return df


    def asignar_split(sem):
        if sem in train_weeks:
            return "train"
        if sem in val_weeks:
            return "val"
        if sem in test_weeks:
            return "test"
        return "out"

    df[SPLIT_COL] = df[WEEK_COL].apply(asignar_split)

    g = df.groupby(WEEK_COL)[SPLIT_COL].nunique()
    assert (g <= 1).all(), "Hay semanas asignadas a más de un split."

    means = df.groupby(SPLIT_COL)[WEEK_COL].mean()
    if {"train", "val", "test"}.issubset(means.index):
        assert means["train"] < means["val"] < means["test"], \
            "Orden temporal train<val<test no se cumple."

    return df


# ==================================================
# 4. ColumnTransformer
# ==================================================

def make_preprocessor(df: pd.DataFrame, use_size_bin: bool = True):
    """
    ColumnTransformer para:
      - semana  -> sin/cos
      - num     -> imputer + IQRClipper + scaler
      - size    -> KBinsDiscretizer (opcional)
      - cat low -> OHE
      - cat hi  -> OrdinalEncoder
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OrdinalEncoder,
        StandardScaler,
        FunctionTransformer,
        KBinsDiscretizer,
    )
    from sklearn.impute import SimpleImputer

    WEEK = [WEEK_COL]

    CAT_LOW = [
        c
        for c in ["customer_type", "sub_category", "segment", "package"]
        if c in df.columns
    ]
    CAT_HIGH = [c for c in ["brand"] if c in df.columns]

    NUM_BASE = [
        c
        for c in [
            "size",
            "num_deliver_per_week",
            "X",
            "Y",
            "compro_semana_pasada",
            "promedio_compra",
            "items_cp",
            "pedidos_cp",
            "items_cliente_semana",
            "items_producto_semana",
            "productos_activos_cliente_semana",
            "recency",
            "customer_buy_rate",
            "product_popularity",
            "score_pop_rec",
        ]
        if c in df.columns
    ]

    if use_size_bin and "size" in NUM_BASE:
        NUM = [c for c in NUM_BASE if c != "size"]
        SIZE_BIN_COLS = ["size"]
    else:
        NUM = NUM_BASE
        SIZE_BIN_COLS = []

    selected = WEEK + NUM + SIZE_BIN_COLS + CAT_LOW + CAT_HIGH
    dups = [c for c in selected if selected.count(c) > 1]
    assert not dups, f"Columnas repetidas entre transformers: {sorted(set(dups))}"

    week_encoder = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("cyc", FunctionTransformer(week_to_sin_cos)),
        ]
    )

    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("clip", IQRClipper(factor=1.5)),
            ("sc", StandardScaler()),
        ]
    )

    if SIZE_BIN_COLS:
        size_bin_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "kb",
                    KBinsDiscretizer(
                        n_bins=4,
                        encode="ordinal",
                        strategy="quantile",
                    ),
                ),
            ]
        )
    else:
        size_bin_pipe = None

    ohe_low = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="UNK")),
            ("ohe", make_ohe(min_freq=None)),
        ]
    )

    ord_high = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="UNK")),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    transformers = [
        ("week", week_encoder, WEEK),
        ("num", num_pipe, NUM),
    ]

    if SIZE_BIN_COLS:
        transformers.append(("sizebin", size_bin_pipe, SIZE_BIN_COLS))
    if CAT_LOW:
        transformers.append(("ohe", ohe_low, CAT_LOW))
    if CAT_HIGH:
        transformers.append(("ord", ord_high, CAT_HIGH))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor


# ==================================================
# 5. Función principal para el DAG (con batch_id)
# ==================================================

def build_weekly_features(
    batch_id: str | None = None,
    save: bool = True,
    max_customers: int | None = 700,
    max_products: int | None = 120,
) -> pd.DataFrame:
    """
    Función pensada para usar desde el DAG de Airflow.

    - Carga datos crudos (clientes, productos, transacciones)
    - (Opcional) limita número de clientes/productos
    - Construye weekly_full_final
    - Limpia columnas irrelevantes
    - Agrega features de comportamiento + extra
    - Crea split temporal
    - Guarda parquet con todas las semanas (incluyendo nuevos batch)
    """
    print(f"[PREPROCESS] build_weekly_features llamado con batch_id={batch_id}")
    print(
        f"[PREPROCESS] Límites: max_customers={max_customers}, max_products={max_products}"
    )

    df_clientes, df_productos, df_transacciones = load_raw_data()

    # ====== LIMITAR TAMAÑO PARA NO REVENTAR MEMORIA ======
    if max_customers is not None and "customer_id" in df_transacciones.columns:
        unique_customers = df_transacciones["customer_id"].drop_duplicates()
        print(f"[PREPROCESS] Clientes únicos en datos crudos: {len(unique_customers)}")
        if len(unique_customers) > max_customers:
            sampled_customers = unique_customers.sample(
                n=max_customers, random_state=42
            )
            df_transacciones = df_transacciones[
                df_transacciones["customer_id"].isin(sampled_customers)
            ]
            df_clientes = df_clientes[
                df_clientes["customer_id"].isin(sampled_customers)
            ]
            print(f"[PREPROCESS] Clientes muestreados: {len(sampled_customers)}")

    if max_products is not None and "product_id" in df_transacciones.columns:
        unique_products = df_transacciones["product_id"].drop_duplicates()
        print(f"[PREPROCESS] Productos únicos en datos crudos: {len(unique_products)}")
        if len(unique_products) > max_products:
            sampled_products = unique_products.sample(
                n=max_products, random_state=42
            )
            df_transacciones = df_transacciones[
                df_transacciones["product_id"].isin(sampled_products)
            ]
            df_productos = df_productos[
                df_productos["product_id"].isin(sampled_products)
            ]
            print(f"[PREPROCESS] Productos muestreados: {len(sampled_products)}")

    # 1) base semanal enriquecida
    df = _build_weekly_full_final(df_clientes, df_productos, df_transacciones)
    print(f"[PREPROCESS] Después de _build_weekly_full_final: {df.shape}")

    # 2) eliminar columnas irrelevantes / constantes
    df = _drop_irrelevant_columns(df)
    print(f"[PREPROCESS] Después de _drop_irrelevant_columns: {df.shape}")

    # 3) agregar variables de comportamiento
    df = _add_behavior_features(df)
    print(f"[PREPROCESS] Después de _add_behavior_features: {df.shape}")

    # 4) features extra
    df = _add_extra_features(df)
    print(f"[PREPROCESS] Después de _add_extra_features: {df.shape}")

    # 5) split temporal
    df = _add_temporal_split(df)
    print(f"[PREPROCESS] Después de _add_temporal_split: {df.shape}")

    # Logging opcional en MLflow
    try:
        import mlflow

        mlflow.log_metric("preprocess_n_rows", len(df))
        if WEEK_COL in df.columns:
            mlflow.log_metric("preprocess_week_min", int(df[WEEK_COL].min()))
            mlflow.log_metric("preprocess_week_max", int(df[WEEK_COL].max()))
    except Exception as e:
        print(f"[PREPROCESS] No se pudieron loguear métricas en MLflow: {e}")

    if save:
        save_features(df)
        print("[PREPROCESS] Features guardadas en FEATURES_PATH")

    return df
