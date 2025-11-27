import kagglehub
import pandas as pd
import os

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Faz o download do dataset da Olist via KaggleHub e carrega o arquivo CSV desejado.
    
    Parâmetros:
    - file_name: nome do arquivo CSV que deseja carregar (ex: 'olist_orders_dataset.csv')
    
    Retorna:
    - DataFrame pandas com os dados carregados
    """
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    file_path = os.path.join(path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo '{file_name}' não encontrado em {path}")
    
    return pd.read_csv(file_path)
