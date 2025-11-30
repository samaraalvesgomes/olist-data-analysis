# Correlação entre Frete e Decisão de Compra

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração visual do matplotlib/seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


try:
    df_orders = pd.read_csv('olist_orders_dataset.csv')
    df_items = pd.read_csv('olist_order_items_dataset.csv')
    df_customers = pd.read_csv('olist_customers_dataset.csv')
    print("Dados carregados com sucesso!")

except FileNotFoundError as e:
    print(f"Erro ao carregar o arquivo: {e}. Por favor, verifique os caminhos.")
   



df_analise = df_orders.merge(df_items[['order_id', 'price', 'freight_value']], 
                             on='order_id', 
                             how='inner')

df_analise = df_analise.merge(df_customers[['customer_id', 'customer_state']], 
                              on='customer_id', 
                              how='left')


df_analise = df_analise[df_analise['order_status'] == 'delivered'].copy()

df_analise.dropna(subset=['price', 'freight_value', 'customer_state'], inplace=True)
df_analise = df_analise[(df_analise['price'] > 0) & (df_analise['freight_value'] >= 0)]



df_analise['preco_total'] = df_analise['price'] + df_analise['freight_value']
df_analise['peso_relativo_frete'] = df_analise['freight_value'] / df_analise['preco_total']

df_agrupado_estado = df_analise.groupby('customer_state').agg(
    volume_pedidos=('order_id', 'nunique'),
    media_prf=('peso_relativo_frete', 'mean'),
    mediana_prf=('peso_relativo_frete', 'median'),
    frete_medio=('freight_value', 'mean')
).reset_index().sort_values(by='volume_pedidos', ascending=False)

total_pedidos = df_agrupado_estado['volume_pedidos'].sum()
df_agrupado_estado['percentual_pedidos'] = (df_agrupado_estado['volume_pedidos'] / total_pedidos) * 100




top_10_estados = df_agrupado_estado.head(10)

fig, ax1 = plt.subplots(figsize=(14, 7))


color = 'tab:blue'
ax1.set_xlabel('Estado do Cliente')
ax1.set_ylabel('Volume de Pedidos (Count)', color=color)
ax1.bar(top_10_estados['customer_state'], top_10_estados['volume_pedidos'], color=color, alpha=0.6, label='Volume de Pedidos')
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Peso Relativo Médio do Frete (PRF)', color=color)
ax2.plot(top_10_estados['customer_state'], top_10_estados['media_prf'], color=color, marker='o', linestyle='--', label='PRF Médio')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, top_10_estados['media_prf'].max() * 1.1) 
plt.title('Volume de Pedidos vs. Peso Relativo do Frete (PRF) - TOP 10 Estados', fontsize=14)
plt.figtext(0.5, 0.01, "O PRF mede a proporção do custo do frete no valor total da compra. Valores mais altos (próximo de 1) indicam que o frete é caro em relação ao produto.", 
            ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
plt.show()


print("\n--- Interpretação e Pontos de Ação ---")
print("Tabela dos TOP 5 Estados:")
print(df_agrupado_estado[['customer_state', 'percentual_pedidos', 'media_prf', 'frete_medio']].head())

media_geral_prf = df_agrupado_estado['media_prf'].mean()
regioes_risco = df_agrupado_estado[df_agrupado_estado['media_prf'] > media_geral_prf].sort_values(by='media_prf', ascending=False)

print(f"\n- Média Geral do PRF em todos os Estados: {media_geral_prf:.2f}")
print("- Estados onde o PRF está acima da média geral (Risco de Inibição de Compra):")
print(regioes_risco[['customer_state', 'media_prf']].head())

