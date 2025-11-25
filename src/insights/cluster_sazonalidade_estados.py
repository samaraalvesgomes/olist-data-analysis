import sys
import os

# Permite imports relativos a partir da raiz do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.common.load_data import load_all_data
from src.common.preprocess import preprocess_data
from src.common.feriados import get_feriados, get_eventos_culturais
from src.common.utils import save_current_fig


def clusterizar_estados_por_sazonalidade(
    df,
    categoria: str,
    ano: int = 2025,
    n_clusters: int = 4,
    save: bool = True,
    show: bool = True
):
    """
    Agrupa estados com padrões de sazonalidade semelhantes usando KMeans
    e marca feriados e eventos culturais no gráfico.
    """

    # Garantir eixo temporal correto
    df["year_month"] = pd.to_datetime(df["year_month"].astype(str))

    # Agregar por categoria, estado e mês
    grouped = (
        df.groupby(["product_category_name", "customer_state", "year_month"])
        .size()
        .reset_index(name="num_orders")
    )

    # Filtrar pela categoria
    subset = grouped[grouped["product_category_name"] == categoria]
    if subset.empty:
        print(f"Nenhum dado encontrado para a categoria: {categoria}")
        return

    # Pivot: estado × mês (valores = número de pedidos)
    pivot = subset.pivot(index="customer_state", columns="year_month", values="num_orders").fillna(0)

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pivot["cluster"] = kmeans.fit_predict(X)

    # Saída no terminal
    print("\n📊 Estados agrupados por padrão de sazonalidade:")
    for cluster_id in range(n_clusters):
        estados = pivot[pivot["cluster"] == cluster_id].index.tolist()
        print(f"Cluster {cluster_id}: {', '.join(estados) if estados else '(vazio)'}")

    # Feriados e eventos culturais
    feriados_df = get_feriados(ano)
    eventos_df = get_eventos_culturais(ano)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for estado in pivot.index:
        serie = pivot.loc[estado].drop("cluster")
        ax.plot(serie.index, serie.values, label=f"{estado} (C{pivot.loc[estado, 'cluster']})")

    # Marcar feriados (linhas verticais)
    for feriado in feriados_df["data"]:
        ax.axvline(feriado, color="red", linestyle="--", alpha=0.25)

    # Marcar eventos culturais
    for evento in eventos_df["data"]:
        ax.axvline(evento, color="purple", linestyle=":", alpha=0.35)

    ax.set_title(f"Clusterização de sazonalidade por estado - categoria '{categoria}'")
    ax.set_xlabel("Ano/Mês")
    ax.set_ylabel("Número de pedidos (normalizado)")

    # Legenda sem duplicatas
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", bbox_to_anchor=(1, 1))

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar e/ou mostrar
    if save:
        save_current_fig("outputs/clusters", f"{categoria}_{ano}_clusters{n_clusters}.png")
        print(f"🖼️ Gráfico salvo em outputs/clusters/{categoria}_{ano}_clusters{n_clusters}.png")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    # Carregar e preparar os dados
    data = load_all_data()
    df = preprocess_data(data)

    # Categoria e parâmetros
    categoria = "beleza_saude"
    ano = 2025
    n_clusters = 4

    # Executar clusterização
    clusterizar_estados_por_sazonalidade(
        df,
        categoria=categoria,
        ano=ano,
        n_clusters=n_clusters,
        save=True,
        show=True
    )


if __name__ == "__main__":
    main()


