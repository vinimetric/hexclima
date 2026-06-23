import os
import numpy as np
import pandas as pd
from fetch_inmet import generate_synthetic_inmet

def generate_synthetic_ana(inmet_path="data/raw/inmet.csv", output_path="data/raw/ana.csv"):
    """
    Gera dados hidrológicos sintéticos (nível do rio e vazão) correlacionados com a chuva.
    Usa um filtro de retardo (lag) para simular o tempo de resposta da bacia hidrográfica.
    """
    if not os.path.exists(inmet_path):
        print(f"Arquivo INMET não encontrado em {inmet_path}. Gerando dados primeiro...")
        df_inmet = generate_synthetic_inmet(output_path=inmet_path)
    else:
        df_inmet = pd.read_csv(inmet_path)
        df_inmet['timestamp'] = pd.to_datetime(df_inmet['timestamp'])
        
    print(f"Gerando dados hidrológicos correlacionados com a chuva em {inmet_path}...")
    precip = df_inmet['precipitacao'].values
    n_hours = len(df_inmet)
    
    # Simulação hidrológica simplificada:
    # Vazão Q(t) acumula chuva com decaimento exponencial (tempo de trânsito da água)
    vazao = np.zeros(n_hours)
    base_vazao = 50.0  # m³/s base
    current_q = base_vazao
    
    # Filtro de resposta hidrológica (lag)
    # Coeficiente de escoamento e taxa de decaimento
    alpha = 0.985  # Decaimento lento (bacia grande)
    beta = 12.0    # Resposta à precipitação
    
    for t in range(n_hours):
        # Vazão é um processo autorregressivo estimulado pela precipitação
        current_q = alpha * current_q + (1 - alpha) * base_vazao + beta * precip[t]
        vazao[t] = current_q
        
    # Adicionar ruído e um pequeno lag adicional para o nível do rio
    # Nível do rio (H) segue uma relação clássica H = c * Q^d (curva de chaveamento)
    # H = base_nivel + coef * Q^0.6 + noise
    nivel_rio = 120.0 + 8.5 * np.power(vazao, 0.58) + np.random.normal(0.0, 3.0, n_hours)
    
    # Ajustando ruído na vazão
    vazao = vazao + np.random.normal(0.0, 1.5, n_hours)
    vazao = np.clip(vazao, 5.0, None)
    nivel_rio = np.clip(nivel_rio, 20.0, None)
    
    df_ana = pd.DataFrame({
        'timestamp': df_inmet['timestamp'],
        'nivel_rio': np.round(nivel_rio, 1),
        'vazao': np.round(vazao, 1)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ana.to_csv(output_path, index=False)
    print(f"Dados hidrológicos salvos com sucesso em {output_path}. Total de registros: {len(df_ana)}")
    return df_ana

if __name__ == "__main__":
    generate_synthetic_ana()
