"""
Arquivo: load_data.py
Descrição: Este módulo contém funções para carregar os datasets da Olist a partir da pasta datasets/raw.
Cada função retorna um DataFrame do Pandas correspondente a uma tabela específica.
Essas funções podem ser utilizadas por todos os membros do projeto para garantir consistência no carregamento dos dados.
"""

import os
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, encoding="utf-8")

def load_all_data() -> dict:
    """
    Carrega os principais datasets da Olist.
    Espera arquivos com nomes padrão dentro de datasets/.
    """
    data = {
        "orders": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_orders_dataset.csv")),
        "order_items": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_order_items_dataset.csv")),
        "customers": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_customers_dataset.csv")),
        "products": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_products_dataset.csv")),
        "sellers": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_sellers_dataset.csv")),
        "geolocation": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_geolocation_dataset.csv")),
        "product_translation": _read_csv_safe(os.path.join(DATASETS_DIR, "product_category_name_translation.csv")),
        # opcional: avaliações e pagamentos, se quiser usar no futuro
        # "reviews": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_order_reviews_dataset.csv")),
        # "payments": _read_csv_safe(os.path.join(DATASETS_DIR, "olist_order_payments_dataset.csv")),
    }
    return data
