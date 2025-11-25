import pandas as pd
from workalendar.america import Brazil

def get_feriados(ano=2025):
    """
    Retorna um DataFrame com os feriados nacionais do Brasil para o ano informado.
    Usa a biblioteca workalendar como fallback.
    """
    cal = Brazil()
    feriados = cal.holidays(ano)  # lista de tuplas (data, nome)

    df = pd.DataFrame(feriados, columns=["data", "nome"])
    df["data"] = pd.to_datetime(df["data"])
    return df


def calcular_black_friday(ano):
    """Última sexta-feira de novembro"""
    nov = pd.date_range(start=f"{ano}-11-01", end=f"{ano}-11-30", freq="D")
    sextas = nov[nov.weekday == 4]
    return sextas[-1]

def calcular_dia_das_maes(ano):
    """Segundo domingo de maio"""
    maio = pd.date_range(start=f"{ano}-05-01", end=f"{ano}-05-31", freq="D")
    domingos = maio[maio.weekday == 6]
    return domingos[1]

def get_eventos_culturais(ano=2025):
    """
    Retorna um DataFrame com eventos culturais extras (não-oficiais).
    """
    eventos = [
        (pd.to_datetime(f"{ano}-06-24"), "São João"),
        (calcular_black_friday(ano), "Black Friday"),
        (calcular_dia_das_maes(ano), "Dia das Mães"),
    ]
    df = pd.DataFrame(eventos, columns=["data", "nome"])
    return df
