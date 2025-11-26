import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================
# Funções comuns embutidas
# ==========================

def ensure_dir(path: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def save_current_fig(path: str, filename: str):
    ensure_dir(path)
    plt.savefig(f"{path}/{filename}", dpi=300)

def load_all_data():
    """Carrega todos os datasets da pasta datasets/"""
    base = "datasets/"
    data = {
        "orders": pd.read_csv(base + "olist_orders_dataset.csv"),
        "order_items": pd.read_csv(base + "olist_order_items_dataset.csv"),
        "customers": pd.read_csv(base + "olist_customers_dataset.csv"),
        "products": pd.read_csv(base + "olist_products_dataset.csv"),
        "sellers": pd.read_csv(base + "olist_sellers_dataset.csv"),
        "geolocation": pd.read_csv(base + "olist_geolocation_dataset.csv"),
        "payments": pd.read_csv(base + "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(base + "olist_order_reviews_dataset.csv"),
        "product_translation": pd.read_csv(base + "product_category_name_translation.csv"),
    }
    return data

def preprocess_data(data: dict) -> pd.DataFrame:
    """Pré-processa os dados e retorna dataframe analítico"""
    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    customers = data["customers"].copy()
    products = data["products"].copy()
    product_translation = data["product_translation"].copy()

    # Garantir datetime
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")

    # Traduzir categorias
    products = products.merge(product_translation, on="product_category_name", how="left")
    products["product_category_name_final"] = products["product_category_name"].fillna(
        products["product_category_name_english"]
    )

    # Merge items com products
    items_products = order_items.merge(
        products[["product_id", "product_category_name_final"]],
        on="product_id",
        how="left"
    )

    # Merge com orders
    items_orders = items_products.merge(
        orders[["order_id", "order_purchase_timestamp", "customer_id"]],
        on="order_id",
        how="left"
    )

    # Merge com customers
    df = items_orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left"
    )

    df = df.rename(columns={"product_category_name_final": "product_category_name"})
    df["year_month"] = pd.to_datetime(df["order_purchase_timestamp"].dt.to_period("M").astype(str))

    return df

def get_feriados(ano: int) -> pd.DataFrame:
    """Retorna alguns feriados nacionais simplificados"""
    datas = [
        {"evento": "Ano Novo", "data": pd.to_datetime(f"{ano}-01-01")},
        {"evento": "Dia do Trabalho", "data": pd.to_datetime(f"{ano}-05-01")},
        {"evento": "Independência", "data": pd.to_datetime(f"{ano}-09-07")},
        {"evento": "Natal", "data": pd.to_datetime(f"{ano}-12-25")},
    ]
    return pd.DataFrame(datas)

def get_eventos_culturais(ano: int) -> pd.DataFrame:
    """Retorna alguns eventos culturais simplificados"""
    datas = [
        {"evento": "Dia das Mães", "data": pd.to_datetime(f"{ano}-05-12")},
        {"evento": "Dia dos Pais", "data": pd.to_datetime(f"{ano}-08-09")},
        {"evento": "Dia dos Namorados", "data": pd.to_datetime(f"{ano}-06-12")},
        {"evento": "Black Friday", "data": pd.to_datetime(f"{ano}-11-29")},
    ]
    return pd.DataFrame(datas)

# ==========================
# Funções de análise
# ==========================

def plot_sazonalidade_por_estado(df, categoria, ano=2025, save=True, export_csv=True):
    df_categoria = df[df["product_category_name"] == categoria]

    if df_categoria.empty:
        print(f"Nenhum dado encontrado para a categoria: {categoria}")
        return

    grouped = (
        df_categoria.groupby(["customer_state", "year_month"])
        .size().reset_index(name="num_orders")
    )

    if export_csv:
        ensure_dir("outputs/sazonalidade")
        grouped.to_csv(f"outputs/sazonalidade/{categoria}_{ano}.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for state in grouped["customer_state"].unique():
        state_data = grouped[grouped["customer_state"] == state]
        ax.plot(state_data["year_month"], state_data["num_orders"], label=state)

    for feriado in get_feriados(ano)["data"]:
        ax.axvline(feriado, color="red", linestyle="--", alpha=0.25)
    for evento in get_eventos_culturais(ano)["data"]:
        ax.axvline(evento, color="purple", linestyle=":", alpha=0.35)

    ax.set_title(f"Tendência da categoria {categoria} por estado ({ano})")
    ax.set_xlabel("Ano/Mês")
    ax.set_ylabel("Número de pedidos")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save:
        save_current_fig("outputs/sazonalidade", f"{categoria}_{ano}.png")
    plt.close()


def clusterizar_estados_por_sazonalidade(df, categoria, ano=2025, n_clusters=4, save=True, export_csv=True):
    df_categoria = df[df["product_category_name"] == categoria]

    if df_categoria.empty:
        print(f"Nenhum dado encontrado para a categoria: {categoria}")
        return

    grouped = (
        df_categoria.groupby(["customer_state", "year_month"])
        .size().reset_index(name="num_orders")
    )

    pivot = grouped.pivot(index="customer_state", columns="year_month", values="num_orders").fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(pivot)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    pivot["cluster"] = kmeans.fit_predict(X)

    if export_csv:
        ensure_dir("outputs/clusters")
        pivot.reset_index()[["customer_state", "cluster"]].to_csv(
            f"outputs/clusters/{categoria}_{ano}_clusters{n_clusters}.csv", index=False
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for estado in pivot.index:
        serie = pivot.loc[estado].drop("cluster")
        ax.plot(serie.index, serie.values, label=f"{estado} (C{pivot.loc[estado, 'cluster']})")

    for feriado in get_feriados(ano)["data"]:
        ax.axvline(feriado, color="red", linestyle="--", alpha=0.25)
    for evento in get_eventos_culturais(ano)["data"]:
        ax.axvline(evento, color="purple", linestyle=":", alpha=0.35)

    ax.set_title(f"Clusterização de sazonalidade - categoria '{categoria}' ({ano})")
    ax.set_xlabel("Ano/Mês")
    ax.set_ylabel("Número de pedidos")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save:
        save_current_fig("outputs/clusters", f"{categoria}_{ano}_clusters{n_clusters}.png")
    plt.close()

# ==========================
# MAIN embutido
# ==========================

def main():
    data = load_all_data()
    df = preprocess_data(data)

    categorias = ["informatica_acessorios", "relogios_presentes", "beleza_saude"]

    for categoria in categorias:
        print(f"\n=== Processando categoria: {categoria} ===")
        plot_sazonalidade_por_estado(df, categoria, ano=2025, save=True, export_csv=True)
        clusterizar_estados_por_sazonalidade(df, categoria, ano=2025, n_clusters=4, save=True, export_csv=True)


if __name__ == "__main__":
    main()
