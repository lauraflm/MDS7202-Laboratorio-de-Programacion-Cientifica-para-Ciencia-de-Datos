# src/preprocessing.py

import numpy as np
import pandas as pd

from .data_io import load_raw_data, save_features


# =====================================================
# 1. Transformadores auxiliares (sin depender de sklearn arriba)
# =====================================================

class IQRClipper:
    """
    Recorta outliers numéricos usando rango intercuartílico (IQR).

    Para cada columna guarda [lo, hi] y luego hace clip en ese rango.
    Compatible con sklearn Pipeline porque implementa fit/transform.
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
    Agrega dos variables de comportamiento por cliente:

    - compro_semana_pasada: 1 si el cliente compró la semana anterior, 0 si no.
    - promedio_compra: promedio histórico del target (compra_flag) hasta la semana anterior.

    Compatible con sklearn Pipeline porque implementa fit/transform.
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
        # No aprende parámetros, solo usa la estructura de X
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
        compras_por_semana["semana_anterior"] = (
            compras_por_semana[self.semana_col] - 1
        )

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
        X_sorted = X_transformed.sort_values(
            [self.customer_col, self.semana_col]
        )

        expanding_mean = (
            X_sorted.groupby(self.customer_col)[self.target_col]
            .expanding()
            .mean()
            .shift(1)  # no incluye la semana actual
            .fillna(0)
            .values
        )

        X_sorted["promedio_compra"] = expanding_mean
        # Volvemos al orden original
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
    Wrapper simple para OneHotEncoder compatible con versiones nuevas/viejas de sklearn.
    Lo importamos luego dentro de make_preprocessor.
    """
    from sklearn.preprocessing import OneHotEncoder  # import local

    try:
        # sklearn >= 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            **({"min_frequency": min_freq} if min_freq is not None else {})
        )
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            **({"min_frequency": min_freq} if min_freq is not None else {})
        )


# ==========================================================
# 2. Construcción de la base semanal (weekly_full_final)
# ==========================================================

def _build_weekly_full_final(df_clientes, df_productos, df_transacciones):
    """
    Construye weekly_full_final (base semanal cliente × producto × semana):

    - Base C×P×semana (todas las semanas presentes en transacciones).
    - items_cp, pedidos_cp, compra_flag.
    - Merge con productos.
    - Merge con clientes.
    """

    # --- base desde transacciones ---
    tx = df_transacciones.copy()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"]).dt.normalize()
    tx["semana"] = tx["purchase_date"].dt.isocalendar().week.astype(int)

    # Aseguramos semanas entre 1 y 53 (por si algún año tiene semana 53)
    tx["semana"] = tx["semana"].clip(1, 53)

    # Agregamos por cliente-producto-semana
    base_cp_sem = (
        tx.groupby(["customer_id", "product_id", "semana"], as_index=False)
        .agg(
            items_cp=("items", "sum"),
            pedidos_cp=("order_id", "nunique"),
        )
        .assign(
            compra_flag=lambda d: (d["items_cp"] > 0).astype("int8")
        )
    )

    # --- Grid completo cliente × producto × semana ---
    weeks_total = np.sort(tx["semana"].unique())
    customers_total = tx["customer_id"].unique()
    products_total = tx["product_id"].unique()

    idx = pd.MultiIndex.from_product(
        [customers_total, products_total, weeks_total],
        names=["customer_id", "product_id", "semana"],
    )
    full = idx.to_frame(index=False)

    # Merge con la base agregada (incluye no-compras)
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

    weekly_full["compra_flag"] = (
        weekly_full["items_cp"] > 0
    ).astype("int8")
    weekly_full["pedidos_cp"] = weekly_full["pedidos_cp"].astype("int16")

    # --- Merge con productos ---
    weekly_enriquecido = weekly_full.merge(
        df_productos,
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    weekly_enriquecido["producto_en_catalogo"] = (
        weekly_enriquecido["brand"].notna()
    )

    # --- Limpieza de df_clientes (X, Y, region_id, zone_id) ---
    df_clientes = df_clientes.copy()
    df_clientes["X"] = pd.to_numeric(df_clientes["X"], errors="coerce")
    df_clientes["Y"] = pd.to_numeric(df_clientes["Y"], errors="coerce")

    for c in ["region_id", "zone_id"]:
        if c in df_clientes.columns:
            df_clientes[c] = pd.to_numeric(
                df_clientes[c], errors="coerce"
            ).astype("Int64")

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


def _add_behavior_features(df):
    """
    Aplica ComportamientoCompraTransformer para agregar:
    - compro_semana_pasada
    - promedio_compra
    """
    from sklearn.pipeline import Pipeline  # import local

    comportamiento_pipeline = Pipeline(
        [
            (
                "comportamiento",
                ComportamientoCompraTransformer(
                    customer_col="customer_id",
                    semana_col=WEEK_COL,
                    target_col=TARGET_COL,
                ),
            )
        ]
    )
    df_with_behavior = comportamiento_pipeline.fit_transform(df)
    return df_with_behavior


def _drop_irrelevant_columns(df):
    """
    Elimina columnas que no aportan o son constantes:

    - producto_en_catalogo
    - category (si tiene 1 solo valor)
    - num_visit_per_week (si es constante)
    """
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
    """
    Crea la columna 'split' usando reglas simples:

    - semanas  1–36 -> train
    - semanas 37–44 -> val
    - semanas 45–53 -> test

    Si alguna semana cae fuera de ese rango, queda como 'out'.
    """
    if SPLIT_COL in df.columns:
        return df

    # Aseguramos semanas entre 1 y 53
    df[WEEK_COL] = df[WEEK_COL].clip(1, 53)

    train_weeks = set(range(1, 37))   # 1–36
    val_weeks = set(range(37, 45))    # 37–44
    test_weeks = set(range(45, 54))   # 45–53 (por si existe semana 53)

    def asignar_split(sem):
        if sem in train_weeks:
            return "train"
        if sem in val_weeks:
            return "val"
        if sem in test_weeks:
            return "test"
        return "out"

    df[SPLIT_COL] = df[WEEK_COL].apply(asignar_split)

    # chequeos
    g = df.groupby(WEEK_COL)[SPLIT_COL].nunique()
    assert (g <= 1).all(), "Hay semanas asignadas a más de un split."

    means = df.groupby(SPLIT_COL)[WEEK_COL].mean()
    if {"train", "val", "test"}.issubset(means.index):
        assert (
            means["train"] < means["val"] < means["test"]
        ), "Orden temporal train<val<test no se cumple."

    return df


# ==================================================
# 4. ColumnTransformer (preprocessor) con imports locales
# ==================================================

def make_preprocessor(df, use_size_bin: bool = True):
    """
    Crea el ColumnTransformer con la misma lógica de la entrega:

    - semana  -> sin/cos
    - num     -> imputer + IQRClipper + scaler
    - size    -> KBinsDiscretizer (opcional)
    - cat low -> OneHotEncoder
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

    # --- Grupos de columnas ---
    WEEK = [WEEK_COL]

    # categóricas baja/alta cardinalidad
    CAT_LOW = [
        c
        for c in ["customer_type", "sub_category", "segment", "package"]
        if c in df.columns
    ]  # OHE
    CAT_HIGH = [
        c for c in ["brand"] if c in df.columns
    ]  # Ordinal

    # numéricas base
    NUM_BASE = [
        c
        for c in [
            "size",
            "num_deliver_per_week",
            "X",
            "Y",
            "compro_semana_pasada",
            "promedio_compra",
        ]
        if c in df.columns
    ]

    if use_size_bin and "size" in NUM_BASE:
        NUM = [c for c in NUM_BASE if c != "size"]
        SIZE_BIN_COLS = ["size"]
    else:
        NUM = NUM_BASE
        SIZE_BIN_COLS = []

    # Chequeo de que ninguna variable quede en dos grupos
    selected = WEEK + NUM + SIZE_BIN_COLS + CAT_LOW + CAT_HIGH
    dups = [c for c in selected if selected.count(c) > 1]
    assert not dups, f"Columnas repetidas entre transformers: {sorted(set(dups))}"

    # --- Pipelines por grupo de columnas ---

    week_encoder = Pipeline(
    [
        ("imp", SimpleImputer(strategy="median")),
        (
            "cyc",
            FunctionTransformer(week_to_sin_cos),
        ),
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

    # OHE lo construimos con make_ohe (que importa sklearn dentro)
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
# 5. Función principal para el DAG (con límite opcional)
# ==================================================

def build_weekly_features(
    save: bool = True,
    max_customers: int | None = 500,
    max_products: int | None = 80,
) -> pd.DataFrame:
    """
    Función pensada para usar desde el DAG de Airflow.

    Hace TODO el preprocesamiento de la Entrega 1:
    - Carga datos crudos (clientes, productos, transacciones)
    - (Opcional) limita número de clientes/productos para no romper memoria
    - Construye weekly_full_final (base semanal C×P×semana)
    - Limpia columnas / tipos
    - Agrega features de comportamiento
    - Crea el split temporal train/val/test
    - (Opcional) guarda el resultado en parquet vía save_features()

    max_customers, max_products permiten que el DAG corra en una máquina
    con poca RAM. Si los quieres todos, pásalos como None.
    """
    df_clientes, df_productos, df_transacciones = load_raw_data()

    # ====== LIMITAR TAMAÑO PARA NO REVENTAR MEMORIA ======
    if max_customers is not None and "customer_id" in df_transacciones.columns:
        unique_customers = df_transacciones["customer_id"].drop_duplicates()
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

    if max_products is not None and "product_id" in df_transacciones.columns:
        unique_products = df_transacciones["product_id"].drop_duplicates()
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
    # =====================================================

    # 1) base semanal enriquecida
    df = _build_weekly_full_final(df_clientes, df_productos, df_transacciones)

    # 2) eliminar columnas irrelevantes / constantes
    df = _drop_irrelevant_columns(df)

    # 3) agregar variables de comportamiento
    df = _add_behavior_features(df)

    # 4) agregar columna 'split' por semana
    df = _add_temporal_split(df)

    if save:
        save_features(df)

    return df
