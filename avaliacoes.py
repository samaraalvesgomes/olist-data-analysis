import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from datetime import date

# Estilo dos gráficos
sns.set(style="whitegrid")

# Dados fictícios
np.random.seed(42)
num_samples = 500

# Datas aleatórias entre 2018-01-01 e 2018-12-31
review_creation_dates = pd.to_datetime(np.random.choice(pd.date_range("2018-01-01", "2018-12-31"), num_samples))
# Tempo de resposta entre 0 e 5 dias
response_deltas = pd.to_timedelta(np.random.randint(0, 6, size=num_samples), unit='d')
review_answer_timestamps = review_creation_dates + response_deltas

# DataFrame
df = pd.DataFrame({
    "review_id": [f"rev_{i}" for i in range(num_samples)],
    "order_id": [f"ord_{i}" for i in range(num_samples)],
    "review_score": np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
    "review_comment_title": np.random.choice(["Ótimo", "Ruim", "Bom", "Péssimo", "Excelente", None], size=num_samples),
    "review_comment_message": np.random.choice([
        "Entrega rápida e produto excelente!",
        "Produto veio com defeito.",
        "Gostei muito, recomendo!",
        "Demorou demais para chegar.",
        "Atendimento maravilhoso!",
        None
    ], size=num_samples),
    "review_creation_date": review_creation_dates,
    "review_answer_timestamp": review_answer_timestamps
})

# Diretório de saída (adaptação para o repositório)
output_dir = os.path.join("plots", "insights", "reviews")
os.makedirs(output_dir, exist_ok=True)

# Feriados Brasileiros 2018
feriados_2018 = {
    date(2018, 1, 1): "Ano Novo",
    date(2018, 2, 13): "Terça de Carnaval",
    date(2018, 3, 29): "Sexta-feira Santa",
    date(2018, 4, 21): "Tiradentes",
    date(2018, 5, 1): "Dia do Trabalho",
    date(2018, 9, 7): "Independência do Brasil",
    date(2018, 10, 12): "Nossa Senhora Aparecida",
    date(2018, 11, 2): "Finados",
    date(2018, 11, 15): "Proclamação da República",
    date(2018, 11, 20): "Consciência Negra",
    date(2018, 12, 25): "Natal",
}

# Adicionar coluna de feriado ao DataFrame
df['is_feriado'] = df['review_creation_date'].dt.date.map(lambda x: x in feriados_2018)
df['feriado_nome'] = df['review_creation_date'].dt.date.map(lambda x: feriados_2018.get(x, None))

# 1. Gráfico de barras da distribuição das notas
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="review_score", hue="review_score", data=df, palette="viridis", dodge=False)

try:
    ax.get_legend().remove()
except Exception:
    pass
plt.title("Distribuição das Notas de Avaliação")
plt.xlabel("Nota")
plt.ylabel("Quantidade")
plt.tight_layout()
bar_chart_path = os.path.join(output_dir, "distribuicao_notas.png")
plt.savefig(bar_chart_path)
plt.close()

# 2. Linha do tempo da média das notas por mês
df['review_month'] = df['review_creation_date'].dt.to_period('M').dt.to_timestamp()
monthly_avg = df.groupby('review_month')['review_score'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x='review_month', y='review_score', data=monthly_avg, marker='o', color='steelblue')
plt.title("Média das Notas por Mês")
plt.xlabel("Mês")
plt.ylabel("Média da Nota")
plt.xticks(rotation=45)
plt.tight_layout()
line_chart_path = os.path.join(output_dir, "media_notas_por_mes.png")
plt.savefig(line_chart_path)
plt.close()

# 3. Histograma do tempo de resposta
df['response_time_days'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.days

plt.figure(figsize=(8, 5))
sns.histplot(df['response_time_days'], bins=range(0, 7), kde=False, color='coral')
plt.title("Tempo de Resposta (em dias)")
plt.xlabel("Dias entre criação e resposta")
plt.ylabel("Número de Avaliações")
plt.tight_layout()
histogram_path = os.path.join(output_dir, "tempo_resposta_histograma.png")
plt.savefig(histogram_path)
plt.close()

print("Gráficos gerados com sucesso: distribuição das notas, média mensal das notas e tempo de resposta.")
