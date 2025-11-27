from src.api.data_loader import load_dataset

def testar_carregamento():
    df = load_dataset("olist_orders_dataset.csv")
    print(df.head())

if __name__ == "__main__":
    testar_carregamento()

