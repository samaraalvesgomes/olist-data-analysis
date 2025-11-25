import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib.pyplot as plt
from src.common.load_data import load_all_data
from src.common.preprocess import preprocess_data
from src.common.feriados import get_feriados, get_eventos_culturais
from src.common.utils import save_current_fig

def plot_sazonalidade_por_estado(df, categoria, ano=2025, save=False):
    df["year_month"] = pd.to_datetime(df["year_month"].astype(str))

    grouped = (
        df.groupby(["product_category_name", "customer_state", "year_month"])
        .size().reset_index(name="num_orders")
    )
    subset = grouped[grouped["product_category_name"] == categoria]
    if subset.empty:
        print(f"Nenhum dado encontrado para a categoria: {categoria}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for state in subset["customer_state"].unique():
        state_data = subset[subset["customer_state"] == state]
        ax.plot(state_data["year_month"], state_data["num_orders"], label=state)

    feriados_df = get_feriados(ano)
    eventos_df = get_eventos_culturais(ano)

    for feriado in feriados_df["data"]:
        ax.axvline(feriado, color="red", linestyle="--", alpha=0.25, label="Feriado")
    for evento in eventos_df["data"]:
        ax.axvline(evento, color="purple", linestyle=":", alpha=0.35, label="Evento cultural")

    ax.set_title(f"Tendência oculta da categoria {categoria} por estado ({ano})")
    ax.set_xlabel("Ano/Mês")
    ax.set_ylabel("Número de pedidos")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if save:
        save_current_fig("outputs/sazonalidade", f"{categoria}_{ano}.png")
    plt.show()

def main():
    data = load_all_data()
    df = preprocess_data(data)
    categoria = "beleza_saude"
    plot_sazonalidade_por_estado(df, categoria, ano=2025, save=True)

if __name__ == "__main__":
    main()
