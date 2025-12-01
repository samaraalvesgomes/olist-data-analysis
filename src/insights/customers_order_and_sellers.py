import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ..api.data_loader import load_dataset
import os

plots_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'plots', 'customers_order_and_sellers')
os.makedirs(plots_path, exist_ok=True)

customers = load_dataset('olist_customers_dataset.csv')
sellers = load_dataset('olist_sellers_dataset.csv')
order_items = load_dataset('olist_order_items_dataset.csv')

customer_state = customers['customer_state'].value_counts(normalize=True) * 100
seller_state = sellers['seller_state'].value_counts(normalize=True) * 100

merged = pd.merge(order_items, sellers, on='seller_id')
avg_by_state = merged.groupby('seller_state')[['price', 'freight_value']].mean()

top_sellers = order_items.groupby('seller_id').size().sort_values(ascending=False).head(10)
top_locations = pd.merge(top_sellers.reset_index(name='items_sold'), sellers, on='seller_id')

print("\n## Top 10 Vendedores por Itens Vendidos (Volume)")
print("----------------------------------------------------------------------")
print(top_sellers.to_string(header=['Itens Vendidos']))
print("----------------------------------------------------------------------")

top_locations['seller_label'] = top_locations['seller_id'].str[:6] + '... (' + top_locations['seller_city'].str.title() + '/' + top_locations['seller_state'] + ')'
top_locations_sorted = top_locations.sort_values('items_sold', ascending=True)

freight_by_state = merged.groupby('seller_state')['freight_value'].mean()

with PdfPages(os.path.join(plots_path, 'relatorio_olist_completo.pdf')) as pdf:
    
    plt.figure(figsize=(10,6))
    customer_state.plot(kind='bar', color='blue')
    plt.title('Distribuição de Clientes por Estado (%)')
    plt.ylabel('% do Total')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    plt.figure(figsize=(10,6))
    seller_state.plot(kind='bar', color='green')
    plt.title('Distribuição de Vendedores por Estado (%)')
    plt.ylabel('% do Total')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    plt.figure(figsize=(10,6))
    avg_by_state['price'].plot(kind='bar', color='orange')
    plt.title('Preço Médio por Estado de Vendedor')
    plt.ylabel('Valor (R$)')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    plt.figure(figsize=(10,6))
    freight_by_state.plot(kind='bar', color='red')
    plt.title('Frete Médio por Estado de Vendedor')
    plt.ylabel('Valor do Frete (R$)')
    plt.tight_layout()
    pdf.savefig(); plt.close()

    plt.figure(figsize=(12,8))
    plt.barh(top_locations_sorted['seller_label'], top_locations_sorted['items_sold'], color='purple')
    plt.title('Top 10 Vendedores por Quantidade de Itens Vendidos (c/ Localização)')
    plt.xlabel('Quantidade de Vendas')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Relatório gerado")