"""
Arquivo: load_data.py
Descrição: Este módulo contém funções para carregar os datasets da Olist a partir da pasta datasets/raw.
Cada função retorna um DataFrame do Pandas correspondente a uma tabela específica.
Essas funções podem ser utilizadas por todos os membros do projeto para garantir consistência no carregamento dos dados.
"""

import pandas as pd
import os

# Define o caminho base para os arquivos de dados brutos
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../datasets/raw")

def load_customers():
    """Carrega o dataset de clientes"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_customers_dataset.csv"))

def load_orders():
    """Carrega o dataset de pedidos"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_orders_dataset.csv"))

def load_order_items():
    """Carrega o dataset de itens dos pedidos"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_order_items_dataset.csv"))

def load_products():
    """Carrega o dataset de produtos"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_products_dataset.csv"))

def load_order_payments():
    """Carrega o dataset de pagamentos dos pedidos"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_order_payments_dataset.csv"))

def load_order_reviews():
    """Carrega o dataset de avaliações dos pedidos"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_order_reviews_dataset.csv"))

def load_sellers():
    """Carrega o dataset de vendedores"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_sellers_dataset.csv"))

def load_geolocation():
    """Carrega o dataset de geolocalização"""
    return pd.read_csv(os.path.join(DATA_PATH, "olist_geolocation_dataset.csv"))

def load_all_data():
    """
    Carrega todos os datasets e retorna em um dicionário.
    Útil para quem quiser ter acesso a todas as tabelas de uma vez.
    """
    return {
        "customers": load_customers(),
        "orders": load_orders(),
        "order_items": load_order_items(),
        "products": load_products(),
        "payments": load_order_payments(),
        "reviews": load_order_reviews(),
        "sellers": load_sellers(),
        "geolocation": load_geolocation(),
    }
