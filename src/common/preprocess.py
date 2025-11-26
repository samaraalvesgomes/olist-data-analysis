"""
Arquivo: preprocess.py
Descrição: Este módulo contém funções de pré-processamento dos datasets da Olist.
Inclui conversão de colunas de datas, criação de colunas auxiliares (ano/mês) e tratamento de valores nulos.
Essas funções podem ser usadas por todos os membros para padronizar a preparação dos dados.
"""

import pandas as pd

def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def preprocess_data(data: dict) -> pd.DataFrame:
    """
    Constrói um dataframe analítico com colunas:
    - order_id, customer_id, customer_state
    - product_id, product_category_name (em português quando disponível)
    - order_purchase_timestamp
    - year_month (primeiro dia do mês)
    """
    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    customers = data["customers"].copy()
    products = data["products"].copy()
    product_translation = data["product_translation"].copy()

    # Garantir datetime
    _ensure_datetime(orders, "order_purchase_timestamp")

    # Traduzir categorias (inglês -> português)
    if "product_category_name_english" in product_translation.columns:
        products = products.merge(
            product_translation,
            on="product_category_name",
            how="left"
        )
        products["product_category_name_final"] = products["product_category_name"].fillna(
            products["product_category_name_english"]
        )
    else:
        products["product_category_name_final"] = products["product_category_name"]

    # Merge items com products (para ter categoria)
    items_products = order_items.merge(
        products[["product_id", "product_category_name_final"]],
        on="product_id",
        how="left"
    )

    # Merge com orders (para ter data)
    items_orders = items_products.merge(
        orders[["order_id", "order_purchase_timestamp", "customer_id"]],
        on="order_id",
        how="left"
    )

    # Merge com customers (para ter estado)
    df = items_orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left"
    )

    # Normalizar nomes de colunas finais
    df = df.rename(columns={"product_category_name_final": "product_category_name"})

    # Criar coluna year_month (usar primeiro dia do mês para index temporal)
    df["year_month"] = pd.to_datetime(df["order_purchase_timestamp"].dt.to_period("M").astype(str))

    # Remover linhas sem categoria/estado/mês
    df = df.dropna(subset=["product_category_name", "customer_state", "year_month"])

    # Padronizar categoria para snake_case simples (sem acentos)
    df["product_category_name"] = (
        df["product_category_name"]
        .str.lower()
        .str.replace("[^A-Za-z0-9_ ]", "", regex=True)
        .str.replace(" ", "_", regex=False)
    )

    return df
