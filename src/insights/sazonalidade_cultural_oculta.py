import sys
import os

# Adiciona a raiz do projeto ao sys.path para permitir imports diretos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib.pyplot as plt
from src.common.load_data import load_all_data
from src.common.preprocess import preprocess_data



def plot_sazonalidade_por_estado(df, categoria):
    # Agrupar pedidos por categoria, estado e mês
    grouped = (
        df.groupby(["product_category_name", "customer_state", "year_month"])
        .size()
        .reset_index(name="num_orders")
    )

    # Filtrar pela categoria desejada
    subset = grouped[grouped["product_category_name"] == categoria]

    if subset.empty:
        print(f"Nenhum dado encontrado para a categoria: {categoria}")
        return

    # Gerar gráfico por estado
    for state in subset["customer_state"].unique():
        state_data = subset[subset["customer_state"] == state]
        plt.plot(state_data["year_month"].astype(str), state_data["num_orders"], label=state)

    plt.xticks(rotation=45)
    plt.title(f"Tendência oculta da categoria {categoria} por estado")
    plt.xlabel("Ano/Mês")
    plt.ylabel("Número de pedidos")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Carregar e preparar os dados
    data = load_all_data()
    df = preprocess_data(data)

    # Escolher a categoria para análise
    categoria = "casa_construcao"  # ajuste conforme necessário

    # Gerar gráfico
    plot_sazonalidade_por_estado(df, categoria)

if __name__ == "__main__":
    main()
