import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.api.data_loader import load_dataset

os.makedirs("plots/insights", exist_ok=True)
sns.set(style="whitegrid")

def _to_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# 11) Margem latente por categoria de produto
def margem_latente_categorias(cost_ratio_default=0.6, price_increase_scenarios=[0.05, 0.10]):
    """
    Agrega por product_category_name e:
      - calcula volume (qty), preço médio, frete médio, peso/volume médios
      - estima custo proxy por categoria usando cost_ratio_default e fatores de peso/volume
      - estima elasticidade preço-demanda por categoria (quando houver variação interna suficiente)
      - simula aumentos de preço por categoria e calcula delta de margem total
    Saídas:
      - CSV com ranking de categorias candidatas
      - Gráfico com top categorias por ganho de margem no cenário +5%
    """
    # Carregar dados
    items = load_dataset("olist_order_items_dataset.csv")
    products = load_dataset("olist_products_dataset.csv")

    # Tipos e limpeza
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce").fillna(0)
    products["product_weight_g"] = pd.to_numeric(products.get("product_weight_g", pd.Series()), errors="coerce")
    products["product_length_cm"] = pd.to_numeric(products.get("product_length_cm", pd.Series()), errors="coerce")
    products["product_height_cm"] = pd.to_numeric(products.get("product_height_cm", pd.Series()), errors="coerce")
    products["product_width_cm"] = pd.to_numeric(products.get("product_width_cm", pd.Series()), errors="coerce")

    # Juntar items + produtos para ter categoria e atributos
    df = items.merge(products, on="product_id", how="left")

    # Agregar por categoria
    cat = df.groupby("product_category_name").agg(
        qty_sold=("order_id", "count"),
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        avg_freight=("freight_value", "mean"),
        avg_weight_g=("product_weight_g", "median"),
        avg_length_cm=("product_length_cm", "median"),
        avg_height_cm=("product_height_cm", "median"),
        avg_width_cm=("product_width_cm", "median")
    ).reset_index().rename(columns={"product_category_name":"category"})

    # Calcular volume médio por categoria (cm3) e preencher NaNs com mediana global
    cat["volume_cm3"] = (cat["avg_length_cm"].fillna(cat["avg_length_cm"].median()) *
                         cat["avg_height_cm"].fillna(cat["avg_height_cm"].median()) *
                         cat["avg_width_cm"].fillna(cat["avg_width_cm"].median()))
    cat["avg_weight_g"] = cat["avg_weight_g"].fillna(cat["avg_weight_g"].median())
    cat["volume_cm3"] = cat["volume_cm3"].fillna(cat["volume_cm3"].median())

    # Normalizar peso/volume para criar cost_factor por categoria
    weight_norm = (cat["avg_weight_g"] - cat["avg_weight_g"].min()) / (cat["avg_weight_g"].max() - cat["avg_weight_g"].min() + 1e-9)
    volume_norm = (cat["volume_cm3"] - cat["volume_cm3"].min()) / (cat["volume_cm3"].max() - cat["volume_cm3"].min() + 1e-9)
    cat["cost_factor"] = 0.9 + 0.6 * (0.5 * weight_norm + 0.5 * volume_norm)  # ~0.9 a ~1.5

    # Estimar custo por unidade (proxy)
    cat["estimated_cost_unit"] = cat["avg_price"] * cost_ratio_default * cat["cost_factor"]

    # Margem atual por unidade e total
    cat["realized_margin_unit"] = cat["avg_price"] - cat["avg_freight"] - cat["estimated_cost_unit"]
    cat["current_revenue_total"] = cat["avg_price"] * cat["qty_sold"]
    cat["current_margin_total"] = cat["realized_margin_unit"] * cat["qty_sold"]

    # Estimar elasticidade preço-demanda por categoria
    # Aqui usamos variação interna entre produtos da mesma categoria: regressão log(qty) ~ log(price)
    # Para isso, agregamos por product_id dentro da categoria
    prod_agg = df.groupby(["product_category_name","product_id"]).agg(
        qty_sold_prod=("order_id","count"),
        avg_price_prod=("price","mean")
    ).reset_index().rename(columns={"product_category_name":"category"})

    elasticity_map = {}
    for category, g in prod_agg.groupby("category"):
        if g.shape[0] < 5:
            elasticity_map[category] = np.nan
            continue
        g = g[(g["avg_price_prod"] > 0) & (g["qty_sold_prod"] > 0)]
        if g.shape[0] < 5:
            elasticity_map[category] = np.nan
            continue
        x = np.log(g["avg_price_prod"].values)
        y = np.log(g["qty_sold_prod"].values)
        try:
            slope = np.polyfit(x, y, 1)[0]
            elasticity_map[category] = float(slope)
        except Exception:
            elasticity_map[category] = np.nan

    cat["price_elasticity"] = cat["category"].map(elasticity_map).fillna(np.nan)

    # Simular cenários de aumento de preço por categoria
    scenario_names = []
    for inc in price_increase_scenarios:
        sname = f"inc_{int(inc*100)}pct"
        scenario_names.append(sname)
        cat[f"sim_price_{sname}"] = cat["avg_price"] * (1 + inc)
        # preencher elasticidade faltante com valor conservador -1.5
        cat["elasticity_filled"] = cat["price_elasticity"].fillna(-1.5)
        # nova quantidade estimada (aprox): qty * (1+inc)^{elasticity}
        cat[f"sim_qty_{sname}"] = (cat["qty_sold"] * ((1 + inc) ** cat["elasticity_filled"])).clip(lower=0)
        cat[f"sim_margin_unit_{sname}"] = cat[f"sim_price_{sname}"] - cat["avg_freight"] - cat["estimated_cost_unit"]
        cat[f"sim_revenue_total_{sname}"] = cat[f"sim_price_{sname}"] * cat[f"sim_qty_{sname}"]
        cat[f"sim_margin_total_{sname}"] = cat[f"sim_margin_unit_{sname}"] * cat[f"sim_qty_{sname}"]
        cat[f"delta_margin_total_{sname}"] = cat[f"sim_margin_total_{sname}"] - cat["current_margin_total"]

    # Ranking por ganho de margem no cenário +5%
    sort_col = "delta_margin_total_inc_5pct" if "delta_margin_total_inc_5pct" in cat.columns else None
    ranking = cat.copy()
    if sort_col:
        ranking = ranking.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    else:
        ranking = ranking.sort_values(by="current_margin_total", ascending=False).reset_index(drop=True)

    # Salvar CSV com colunas relevantes
    out_cols = [
        "category","qty_sold","avg_price","avg_freight","estimated_cost_unit",
        "realized_margin_unit","current_revenue_total","current_margin_total","price_elasticity"
    ]
    for s in scenario_names:
        out_cols += [f"sim_price_{s}", f"sim_qty_{s}", f"sim_margin_unit_{s}", f"sim_margin_total_{s}", f"delta_margin_total_{s}"]

    ranking[out_cols].to_csv("plots/insights/margem_latente_categorias_ranking.csv", index=False)

    # Plot: top 15 categorias por ganho de margem no cenário +5%
    if "delta_margin_total_inc_5pct" in ranking.columns:
        top = ranking.head(15)
        plt.figure(figsize=(12,8))
        sns.barplot(
            x="delta_margin_total_inc_5pct",
            y="category",
            data=top,
            palette="coolwarm"
        )
        plt.xlabel("Ganho estimado de margem total (R$) — +5% preço")
        plt.ylabel("Categoria")
        plt.title("Top 15 categorias com maior ganho de margem estimado (+5% preço)")
        plt.tight_layout()
        plt.savefig("plots/insights/margem_latente_categorias_top15_inc5pct.png")
        print("✅ Gráfico 'margem_latente_categorias_top15_inc5pct' salvo.")
    else:
        print("⚠️ Cenário +5% não disponível para plotagem.")

    print("✅ CSV 'margem_latente_categorias_ranking.csv' salvo.")
    return ranking

# 6) Segmento sensível a parcelas e LTV (mantido)
def parcelas_e_ltv():
    orders = load_dataset("olist_orders_dataset.csv")
    payments = load_dataset("olist_order_payments_dataset.csv")
    customers = load_dataset("olist_customers_dataset.csv")

    orders = _to_datetime(orders, ["order_purchase_timestamp", "order_delivered_customer_date"])
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce").fillna(0)
    payments["payment_installments"] = pd.to_numeric(payments["payment_installments"], errors="coerce").fillna(1)

    payments_orders = payments.merge(orders[["order_id","customer_id","order_purchase_timestamp"]], on="order_id", how="left")
    ltv_by_customer = payments_orders.groupby("customer_id").agg(
        total_revenue=("payment_value", "sum"),
        n_orders=("order_id", lambda x: x.nunique()),
        avg_installments=("payment_installments", "mean"),
        max_installments=("payment_installments", "max")
    ).reset_index()

    def seg_install(x):
        if x <= 1:
            return "sem_parcelas"
        if 1 < x <= 3:
            return "parcelas_baixas_2_3"
        if 3 < x <= 6:
            return "parcelas_medias_4_6"
        return "parcelas_altas_7_plus"

    ltv_by_customer["install_segment"] = ltv_by_customer["avg_installments"].round().apply(seg_install)

    seg_summary = ltv_by_customer.groupby("install_segment").agg(
        clientes=("customer_id", "count"),
        avg_ltv=("total_revenue", "mean"),
        median_ltv=("total_revenue", "median"),
        avg_orders=("n_orders", "mean")
    ).reset_index()

    plt.figure(figsize=(10,6))
    sns.barplot(x="install_segment", y="avg_ltv", data=seg_summary, palette="magma")
    plt.title("LTV médio por segmento de parcelas")
    plt.xlabel("Segmento de parcelas")
    plt.ylabel("LTV médio (R$)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("plots/insights/parcelas_e_ltv.png")
    seg_summary.to_csv("plots/insights/parcelas_e_ltv_summary.csv", index=False)
    print("✅ Gráfico 'parcelas_e_ltv' e CSV salvos.")
    return seg_summary

# 9) Micro‑mercados por zip prefix (mantido)
def micro_mercados_zip():
    orders = load_dataset("olist_orders_dataset.csv")
    items = load_dataset("olist_order_items_dataset.csv")
    customers = load_dataset("olist_customers_dataset.csv")
    geoloc = load_dataset("olist_geolocation_dataset.csv")

    orders = _to_datetime(orders, ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce").fillna(0)

    ticket = items.groupby("order_id").agg(order_value=("price", "sum")).reset_index()
    orders_ticket = orders.merge(ticket, on="order_id", how="left").merge(customers[["customer_id","customer_zip_code_prefix"]], on="customer_id", how="left")

    orders_ticket["delivery_time_days"] = (orders_ticket["order_delivered_customer_date"] - orders_ticket["order_purchase_timestamp"]).dt.days

    zip_summary = orders_ticket.groupby("customer_zip_code_prefix").agg(
        pedidos_count=("order_id", "count"),
        avg_ticket=("order_value", "mean"),
        median_ticket=("order_value", "median"),
        avg_delivery_days=("delivery_time_days", "mean"),
        unique_customers=("customer_id", lambda x: x.nunique())
    ).reset_index()

    geoloc_med = geoloc.groupby("geolocation_zip_code_prefix").agg(
        lat_med=("geolocation_lat", "median"),
        lng_med=("geolocation_lng", "median")
    ).reset_index().rename(columns={"geolocation_zip_code_prefix":"customer_zip_code_prefix"})

    try:
        zip_summary["customer_zip_code_prefix"] = zip_summary["customer_zip_code_prefix"].astype(int)
        geoloc_med["customer_zip_code_prefix"] = geoloc_med["customer_zip_code_prefix"].astype(int)
    except Exception:
        pass

    zip_map = zip_summary.merge(geoloc_med, on="customer_zip_code_prefix", how="left")

    threshold_pedidos = max(50, int(zip_map["pedidos_count"].quantile(0.75)))
    candidates = zip_map[(zip_map["pedidos_count"] >= threshold_pedidos)].copy()
    candidates = candidates.sort_values("pedidos_count", ascending=False)

    plt.figure(figsize=(10,8))
    sc = plt.scatter(
        candidates["lng_med"], candidates["lat_med"],
        s=candidates["pedidos_count"] / candidates["pedidos_count"].max() * 1000 + 50,
        c=candidates["avg_ticket"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="k"
    )
    plt.colorbar(sc, label="Ticket médio (R$)")
    plt.title("Micro‑mercados candidatos (zip prefixes) — tamanho ~ pedidos, cor ~ ticket médio")
    plt.xlabel("Longitude (mediana)")
    plt.ylabel("Latitude (mediana)")
    plt.tight_layout()
    plt.savefig("plots/insights/micro_mercados_zip.png")
    candidates.to_csv("plots/insights/micro_mercados_candidates.csv", index=False)
    print("✅ Gráfico 'micro_mercados_zip' e CSV salvos.")
    return candidates

if __name__ == "__main__":
    print("Iniciando análises: margem latente por categoria (11), parcelas e LTV (6), micro‑mercados (9).")
    ranking_cat = margem_latente_categorias()
    seg_summary = parcelas_e_ltv()
    candidates = micro_mercados_zip()
    print("✅ Todas as análises concluídas. Resultados salvos em plots/insights.")
