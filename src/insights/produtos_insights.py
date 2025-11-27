import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from src.api.data_loader import load_dataset

os.makedirs("plots/produtos-insights", exist_ok=True)

def fotos_vs_vendas():
    df_items = load_dataset("olist_order_items_dataset.csv")
    df_products = load_dataset("olist_products_dataset.csv")
    df = df_items.merge(df_products, on="product_id", how="left")

    vendas_fotos = df.groupby("product_photos_qty")["order_id"].count()

    plt.figure(figsize=(10,6))
    sns.barplot(x=vendas_fotos.index, y=vendas_fotos.values, palette="rocket", edgecolor=None)
    plt.title("Quantidade de fotos vs. Vendas", fontsize=16, fontweight="bold")
    plt.xlabel("Número de fotos do produto", fontsize=12)
    plt.ylabel("Quantidade de pedidos", fontsize=12)

    for i, v in enumerate(vendas_fotos.values):
        plt.text(vendas_fotos.index[i], v + 500, str(v), ha='center', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/produtos-insights/fotos_vs_vendas.png")
    print("✅ Gráfico 'fotos_vs_vendas' salvo.")

def categorias_preco_vendas():
    df_items = load_dataset("olist_order_items_dataset.csv")
    df_products = load_dataset("olist_products_dataset.csv")
    df = df_items.merge(df_products, on="product_id", how="left")

    vendas_categoria = df.groupby("product_category_name")["order_id"].count()
    preco_categoria = df.groupby("product_category_name")["price"].mean()

    top_categorias = vendas_categoria.sort_values(ascending=False).head(8)
    preco_top = preco_categoria[top_categorias.index]

    fig, ax1 = plt.subplots(figsize=(10,6))

    sns.barplot(
        x=top_categorias.values,
        y=top_categorias.index,
        ax=ax1,
        palette="flare"
    )
    ax1.set_xlabel("Quantidade de pedidos", fontsize=12)
    ax1.set_ylabel("Categoria", fontsize=12)

    for i, v in enumerate(top_categorias.values):
        ax1.text(v + 300, i, str(v), va='center', fontsize=9)

    ax2 = ax1.twiny()
    sns.lineplot(
        x=preco_top.values,
        y=preco_top.index,
        ax=ax2,
        color="#d62728",
        marker="o"
    )
    ax2.set_xlabel("Preço médio (R$)", fontsize=12)

    for i, v in enumerate(preco_top.values):
        ax2.text(v + 2, preco_top.index[i], f"R${v:.2f}", va='center', fontsize=9, color="#d62728")

    plt.title("Top categorias mais vendidas e preço médio", fontsize=16, fontweight="bold")

    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("plots/produtos-insights/categorias_preco_vendas.png")
    print("✅ Gráfico 'categorias_preco_vendas' salvo.")

def tamanho_vs_vendas():
    df_items = load_dataset("olist_order_items_dataset.csv")
    df_products = load_dataset("olist_products_dataset.csv")
    df = df_items.merge(df_products, on="product_id", how="left")

    df["product_volume"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    vendas_volume = df.groupby("product_volume")["order_id"].count().reset_index()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=vendas_volume, x="product_volume", y="order_id", alpha=0.6, color="#1f77b4")
    plt.title("Tamanho do produto vs. Vendas", fontsize=16, fontweight="bold")
    plt.xlabel("Volume do produto (cm³)", fontsize=12)
    plt.ylabel("Quantidade de pedidos", fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/produtos-insights/tamanho_vs_vendas.png")
    print("✅ Gráfico 'tamanho_vs_vendas' salvo.")

def descricao_vs_vendas():
    df_items = load_dataset("olist_order_items_dataset.csv")
    df_products = load_dataset("olist_products_dataset.csv")
    df = df_items.merge(df_products, on="product_id", how="left")

    vendas_descricao = df.groupby("product_description_lenght")["order_id"].count().reset_index()

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=vendas_descricao, x="product_description_lenght", y="order_id", alpha=0.6, color="#ff7f0e")
    plt.title("Descrição do produto vs. Vendas", fontsize=16, fontweight="bold")
    plt.xlabel("Comprimento da descrição do produto", fontsize=12)
    plt.ylabel("Quantidade de pedidos", fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/produtos-insights/descricao_vs_vendas.png")
    print("✅ Gráfico 'descricao_vs_vendas' salvo.")

def dashboard_produtos():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    img1 = mpimg.imread("plots/produtos-insights/fotos_vs_vendas.png")
    img2 = mpimg.imread("plots/produtos-insights/categorias_preco_vendas.png")
    img3 = mpimg.imread("plots/produtos-insights/tamanho_vs_vendas.png")
    img4 = mpimg.imread("plots/produtos-insights/descricao_vs_vendas.png")

    axs[0, 0].imshow(img1)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img2)
    axs[0, 1].axis('off')

    axs[1, 0].imshow(img3)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(img4)
    axs[1, 1].axis('off')

    plt.suptitle("ANÁLISE DE PRODUTOS – OLIST", fontsize=20, fontweight="bold")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # aumenta espaçamento entre gráficos
    plt.savefig("plots/produtos-insights/dashboard_produtos.png")
    print("✅ Dashboard final com espaçamento maior salva.")

if __name__ == "__main__":
    fotos_vs_vendas()
    categorias_preco_vendas()
    tamanho_vs_vendas()
    descricao_vs_vendas()
    dashboard_produtos()
