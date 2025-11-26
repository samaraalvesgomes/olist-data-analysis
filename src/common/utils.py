"""
Arquivo: utils.py
Descrição: Este módulo contém funções auxiliares para visualização e manipulação dos dados.
Inclui gráficos de tendência por categoria, que podem ser usados para análises de sazonalidade.
"""

import os
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_current_fig(folder: str, filename: str):
    ensure_dir(folder)
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figura salva em: {out_path}")
