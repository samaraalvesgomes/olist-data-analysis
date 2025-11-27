import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Definir caminho base
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
datasets_path = os.path.join(base_path, 'datasets', 'raw')

# --- 1. CARREGAMENTO DOS DADOS ---
# Lendo os arquivos que você enviou
df_sellers = pd.read_csv(os.path.join(datasets_path, 'olist_sellers_dataset.csv'))
df_items = pd.read_csv(os.path.join(datasets_path, 'olist_order_items_dataset.csv'))
df_orders = pd.read_csv(os.path.join(datasets_path, 'olist_orders_dataset.csv'))
df_products = pd.read_csv(os.path.join(datasets_path, 'olist_products_dataset.csv'))

# --- 2. TRATAMENTO E CRUZAMENTO DE DADOS  ---

# Cruzamento 1: Itens com Pedidos (para ter datas) e Sellers (para ter estados)
df_merged = df_items.merge(df_orders, on='order_id')
df_merged = df_merged.merge(df_sellers, on='seller_id')
df_merged = df_merged.merge(df_products, on='product_id')

# Calculando tempo de entrega (Data Entrega - Data Compra)
df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])
df_merged['delivery_days'] = (df_merged['order_delivered_customer_date'] - df_merged['order_purchase_timestamp']).dt.days

# Limpando dados (removendo entregas não finalizadas ou erros de data negativa)
df_clean = df_merged.dropna(subset=['delivery_days'])
df_clean = df_clean[df_clean['delivery_days'] > 0]

# Configuração visual
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(24, 18))
fig.suptitle('ANÁLISE DE VENDEDORES - OLIST', fontsize=20, fontweight='bold', y=0.995)

# --- 3. GERAÇÃO DOS GRÁFICOS ---

# GRÁFICO 1: Distribuição Geográfica (Onde estão os sellers?)
plt.subplot(2, 2, 1)
seller_state_counts = df_sellers['seller_state'].value_counts().reset_index()
seller_state_counts.columns = ['Estado', 'Qtd Sellers']
sns.barplot(x='Qtd Sellers', y='Estado', data=seller_state_counts.head(10), palette='viridis')
plt.title('1. Top 10 Estados com Maior Concentração de Sellers')
plt.xlabel('Quantidade de Sellers')

# GRÁFICO 2: Volume de Vendas por Estado (Ranking Maiores vs Menores)
plt.subplot(2, 2, 2)
# Contar quantidade de pedidos ÚNICOS (order_id) por estado do seller
state_order_volume = df_clean.groupby('seller_state')['order_id'].nunique().sort_values(ascending=False)
# Pegar Top 10 estados por volume de pedidos
top_10_states = state_order_volume.head(10).reset_index()
top_10_states.columns = ['estado', 'num_orders']
top_10_states = top_10_states.sort_values('num_orders')

# Barplot horizontal com gradiente de cores
sns.barplot(x='num_orders', y='estado', data=top_10_states, palette='rocket_r')
plt.title('2. Volume de Vendas: Top 10 Estados por Número de Pedidos')
plt.xlabel('Número de Pedidos (Order ID Únicos)')
plt.ylabel('Estado')
plt.tight_layout()

# GRÁFICO 3: Categorias por Seller (Nicho Principal)
plt.subplot(2, 2, 3)
# Descobrir qual a categoria PRINCIPAL de cada seller (a que ele mais vende)
seller_main_cat = df_clean.groupby(['seller_id', 'product_category_name']).size().reset_index(name='count')
seller_main_cat = seller_main_cat.sort_values('count', ascending=False).drop_duplicates('seller_id')
# Contar quantos sellers se dedicam a cada nicho
niche_counts = seller_main_cat['product_category_name'].value_counts().head(10)

sns.barplot(x=niche_counts.values, y=niche_counts.index, palette='mako')
plt.title('3. Top 10 Nichos: Categorias Principais dos Sellers')
plt.xlabel('Quantidade de Sellers Dedicados ao Nicho')

# GRÁFICO 4: Correlação Localização vs. Tempo de Entrega
plt.subplot(2, 2, 4)
# Calcular mediana de entrega por estado para ordenar o gráfico
state_order = df_clean.groupby('seller_state')['delivery_days'].median().sort_values().index
# Filtrar apenas estados com volume relevante para o gráfico não ficar poluído (Top 15 estados)
top_states = df_clean['seller_state'].value_counts().head(15).index
df_logistics = df_clean[df_clean['seller_state'].isin(top_states)]

sns.boxplot(x='seller_state', y='delivery_days', data=df_logistics, order=state_order, showfliers=False, palette='coolwarm')
plt.title('4. Gargalos Logísticos: Tempo de Entrega por Estado de Origem')
plt.ylabel('Dias para Entrega (Mediana e Variação)')
plt.xlabel('Estado de Origem do Seller')

# Em execuções automatizadas (CI / headless) salvamos a figura em vez de abrir janelas
output_dir = os.path.join(base_path, 'outputs', 'insights')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'vendedores_summary.png')
try:
	plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.0)
	fig.savefig(output_file, bbox_inches='tight', dpi=150)
	print(f"Gráfico salvo em: {output_file}")
except Exception as e:
	print(f"Erro ao salvar figura: {e}")

# Tenta mostrar interativamente se houver display (não falha em headless)
try:
	plt.show()
except Exception:
	plt.close(fig)