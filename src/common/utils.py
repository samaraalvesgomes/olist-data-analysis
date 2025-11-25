"""
Arquivo: utils.py
Descrição: Este módulo contém funções auxiliares para visualização e manipulação dos dados.
Inclui gráficos de tendência por categoria, que podem ser usados para análises de sazonalidade.
"""

import matplotlib.pyplot as plt
import os


def ensure_dir(path: str):
    """
    Cria o diretório se ele não existir.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_current_fig(path: str, filename: str, dpi: int = 120):
    """
    Salva o gráfico atual (plt) em um arquivo PNG dentro do diretório especificado.
    """
    ensure_dir(path)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"🖼️ Gráfico salvo em {full_path}")


def plot_category_trends(df, category_col="product_category_name", date_col="order_purchase_timestamp"):
    """
    Plota a evolução mensal de pedidos por categoria.
    - df: DataFrame contendo pedidos e produtos
    - category_col: coluna que identifica a categoria do produto
    - date_col: coluna que contém a data da compra
    """
    # Cria coluna combinando ano e mês
    df["year_month"] = df[date_col].dt.to_period("M")

    # Agrupa por categoria e mês, contando número de pedidos
    grouped = df.groupby([category_col, "year_month"]).size().reset_index(name="num_orders")

    # Para cada categoria, plota uma linha mostrando a evolução ao longo do tempo
    for category in grouped[category_col].unique():
        subset = grouped[grouped[category_col] == category]
        plt.plot(subset["year_month"].astype(str), subset["num_orders"], label=category)

    # Ajustes visuais do gráfico
    plt.xticks(rotation=45)
    plt.title("Tendência de pedidos por categoria")
    plt.xlabel("Ano/Mês")
    plt.ylabel("Número de pedidos")
    plt.legend()
    plt.tight_layout()
    plt.show()
