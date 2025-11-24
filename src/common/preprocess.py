"""
Arquivo: preprocess.py
Descrição: Este módulo contém funções de pré-processamento dos datasets da Olist.
Inclui conversão de colunas de datas, criação de colunas auxiliares (ano/mês) e tratamento de valores nulos.
Essas funções podem ser usadas por todos os membros para padronizar a preparação dos dados.
"""

import pandas as pd

def parse_dates(df, date_columns):
    """
    Converte colunas de datas para o tipo datetime do Pandas.
    - df: DataFrame
    - date_columns: lista de nomes de colunas que devem ser convertidas
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def preprocess_orders(orders_df):
    """
    Pré-processa o dataset de pedidos:
    - Converte colunas de datas
    - Cria colunas auxiliares de ano e mês
    """
    orders_df = parse_dates(
        orders_df,
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    # Cria colunas de ano e mês a partir da data de compra
    orders_df["year"] = orders_df["order_purchase_timestamp"].dt.year
    orders_df["month"] = orders_df["order_purchase_timestamp"].dt.month
    return orders_df

def preprocess_products(products_df):
    """
    Pré-processa o dataset de produtos:
    - Preenche valores nulos na coluna de categoria com 'unknown'
    """
    products_df["product_category_name"].fillna("unknown", inplace=True)
    return products_df
