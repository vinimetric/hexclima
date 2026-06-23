import os
import numpy as np
import pandas as pd
from datetime import datetime

def generate_synthetic_inmet(start_date="2018-01-01", end_date="2024-06-30", output_path="data/raw/inmet.csv"):
    """
    Gera dados meteorológicos sintéticos realistas para o Rio Grande do Sul.
    Inclui ciclos diários, sazonais e insere eventos extremos para validação.
    """
    print(f"Gerando dados meteorológicos sintéticos de {start_date} até {end_date}...")
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    n_hours = len(timestamps)
    
    # 1. Temperatura (°C): Ciclo anual + ciclo diário + ruído
    # RS é mais frio em Jun/Jul e mais quente em Jan/Fev
    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values
    
    temp_annual = 18.0 + 8.0 * np.cos(2.0 * np.pi * (day_of_year - 20) / 365.0)
    temp_daily = 5.0 * np.cos(2.0 * np.pi * (hour_of_day - 14) / 24.0)
    temp_noise = np.random.normal(0.0, 2.0, n_hours)
    temperatura = temp_annual + temp_daily + temp_noise
    
    # 2. Umidade relativa (%): Inversamente proporcional à temperatura + ruído
    umidade = 75.0 - 1.2 * temp_daily + np.random.normal(0.0, 5.0, n_hours)
    umidade = np.clip(umidade, 20.0, 100.0)
    
    # 3. Pressão atmosférica (hPa): Média ~1013, levemente menor em dias quentes/tempestivos
    pressao = 1013.0 - 0.1 * temp_annual + np.random.normal(0.0, 3.0, n_hours)
    
    # 4. Velocidade do vento (km/h): Média ~8 km/h com rajadas aleatórias
    velocidade_vento = np.random.exponential(scale=6.0, size=n_hours) + 3.0
    
    # 5. Precipitação acumulada (mm)
    # Normalmente seco. Criamos sistemas frontais e tempestades.
    precipitacao = np.zeros(n_hours)
    
    # Geramos chuva regular (ocorrência a cada ~60 horas em média, durando 3-8 horas)
    np.random.seed(42)  # Seed para consistência
    i = 0
    while i < n_hours:
        # Probabilidade de começar a chover
        if np.random.rand() < 0.015:
            duration = np.random.randint(3, 12)
            intensity = np.random.exponential(scale=2.0)
            for d in range(duration):
                if i + d < n_hours:
                    # Distribuição de chuva com formato de sino
                    precipitacao[i+d] = intensity * (1.0 + np.sin(np.pi * d / duration)) * np.random.uniform(0.5, 1.5)
            i += duration
        else:
            i += 1
            
    # Injetando eventos extremos (anomalias):
    # Evento 1: Enchente Vale do Taquari (Setembro 2023) -> Chuva pesada persistente
    print("Injetando evento extremo: Setembro/2023...")
    mask_sept_2023 = (timestamps >= '2023-09-02 00:00:00') & (timestamps <= '2023-09-06 23:00:00')
    hours_sept = np.sum(mask_sept_2023)
    if hours_sept > 0:
        # Chuva torrencial acumulando ~250-400mm no período
        precipitacao[mask_sept_2023] += np.random.exponential(scale=8.0, size=hours_sept) + 4.0
        velocidade_vento[mask_sept_2023] += np.random.uniform(15, 35, size=hours_sept)
        pressao[mask_sept_2023] -= np.random.uniform(15, 25, size=hours_sept)
        umidade[mask_sept_2023] = np.clip(umidade[mask_sept_2023] + 15, 90.0, 100.0)
        
    # Evento 2: Grande Enchente RS (Maio 2024) -> Chuva catastrófica contínua por semanas
    print("Injetando evento extremo catastrófico: Maio/2024...")
    mask_may_2024 = (timestamps >= '2024-04-28 00:00:00') & (timestamps <= '2024-05-12 23:00:00')
    hours_may = np.sum(mask_may_2024)
    if hours_may > 0:
        # Acumulado gigantesco
        precipitacao[mask_may_2024] += np.random.exponential(scale=12.0, size=hours_may) + 5.0
        velocidade_vento[mask_may_2024] += np.random.uniform(20, 45, size=hours_may)
        pressao[mask_may_2024] -= np.random.uniform(20, 35, size=hours_may)
        umidade[mask_may_2024] = np.clip(umidade[mask_may_2024] + 20, 95.0, 100.0)
        
    # Garantir clipping de precipitação
    precipitacao = np.round(precipitacao, 2)
    velocidade_vento = np.round(velocidade_vento, 2)
    temperatura = np.round(temperatura, 1)
    umidade = np.round(umidade, 1)
    pressao = np.round(pressao, 1)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'precipitacao': precipitacao,
        'velocidade_vento': velocidade_vento,
        'temperatura': temperatura,
        'umidade': umidade,
        'pressao': pressao
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dados salvos com sucesso em {output_path}. Total de registros: {len(df)}")
    return df

if __name__ == "__main__":
    generate_synthetic_inmet()
