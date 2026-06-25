import os
import pickle
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Se executado diretamente ou importado, garantir caminhos relativos consistentes
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def handle_missing(df, max_interp_hours=3, max_ffill_hours=6):
    """
    Estratégia escalonada para dados faltantes.
    """
    df = df.copy()
    # Interpolação linear para lacunas curtas
    df = df.interpolate(method='linear', limit=max_interp_hours)
    # Forward fill para lacunas um pouco maiores
    try:
        df = df.ffill(limit=max_ffill_hours)
    except (TypeError, AttributeError):
        df = df.fillna(method='ffill', limit=max_ffill_hours)
    return df

def create_sequences(df, features, window=72, stride=1):
    """
    Cria sequências temporais de formato (N, window, n_features).
    """
    data = df[features].values
    X = []
    for i in range(0, len(data) - window + 1, stride):
        X.append(data[i:i + window])
    return np.array(X)

def run_preprocessing():
    config = load_config()
    
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    window = config['model']['timesteps']
    scaler_path = config['model']['scaler_path']
    
    inmet_file = os.path.join(raw_dir, "inmet.csv")
    ana_file = os.path.join(raw_dir, "ana.csv")
    
    if not os.path.exists(inmet_file) or not os.path.exists(ana_file):
        raise FileNotFoundError("Os arquivos brutos inmet.csv e ana.csv precisam ser gerados primeiro.")
        
    print("Carregando arquivos de dados brutos...")
    df_inmet = pd.read_csv(inmet_file)
    df_ana = pd.read_csv(ana_file)
    
    df_inmet['timestamp'] = pd.to_datetime(df_inmet['timestamp'])
    df_ana['timestamp'] = pd.to_datetime(df_ana['timestamp'])
    
    print("Mesclando dados meteorológicos e hidrológicos...")
    df_merged = pd.merge(df_inmet, df_ana, on='timestamp', how='outer').sort_values('timestamp')

    # --- Engenharia de Features ---
    print("Criando features de acumulados e médias móveis...")
    # Precipitação acumulada nas últimas 24h
    if 'precipitacao' in df_merged.columns:
        df_merged['precip_24h'] = df_merged['precipitacao'].rolling(window=24).sum()

    # Média móvel do nível do rio nas últimas 48h (tendência)
    if 'nivel_rio' in df_merged.columns:
        df_merged['nivel_rio_ma_48h'] = df_merged['nivel_rio'].rolling(window=48).mean()

    # Atualiza a lista de features para incluir as novas
    # Note: Isso assume que config['data']['features'] será atualizado no YAML
    # Mas aqui garantimos que as novas colunas existam antes de handle_missing

    # Tratamento de valores ausentes
    df_merged[features] = handle_missing(df_merged[features])
    
    # Divisão temporal:
    # Treino: 2018-01-01 a 2022-12-31 (~5 anos de dados normais)
    # Validação: 2023-01-01 a 2023-12-31 (Inclui o evento de Set/2023)
    # Teste: 2024-01-01 a 2024-06-30 (Inclui o evento de Mai/2024)
    
    df_train = df_merged[df_merged['timestamp'] < '2023-01-01'].copy()
    df_val = df_merged[(df_merged['timestamp'] >= '2023-01-01') & (df_merged['timestamp'] < '2024-01-01')].copy()
    df_test = df_merged[df_merged['timestamp'] >= '2024-01-01'].copy()
    
    print(f"Registros - Treino: {len(df_train)}, Validação: {len(df_val)}, Teste: {len(df_test)}")
    
    # Normalização robusta baseada no treino
    print("Normalizando dados usando RobustScaler...")
    scaler = RobustScaler()
    
    # Fit apenas no treino
    df_train[features] = scaler.fit_transform(df_train[features])
    df_val[features] = scaler.transform(df_val[features])
    df_test[features] = scaler.transform(df_test[features])
    
    # Salvar o scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"RobustScaler salvo em {scaler_path}")
    
    # Salvar dados processados
    os.makedirs(processed_dir, exist_ok=True)
    df_train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    # Gerar os rótulos de validação (y_true) para avaliação posterior baseada em timestamps conhecidos
    # O evento de Setembro de 2023 ocorreu de 2 a 10 de Setembro
    # O evento de Maio de 2024 ocorreu de 28 de Abril a 20 de Maio
    df_val['is_anomaly'] = 0
    df_val.loc[(df_val['timestamp'] >= '2023-09-02 00:00:00') & (df_val['timestamp'] <= '2023-09-10 23:00:00'), 'is_anomaly'] = 1
    
    df_test['is_anomaly'] = 0
    df_test.loc[(df_test['timestamp'] >= '2024-04-28 00:00:00') & (df_test['timestamp'] <= '2024-05-20 23:00:00'), 'is_anomaly'] = 1
    
    # Salvar com os rótulos
    df_val.to_csv(os.path.join(processed_dir, "val_labeled.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "test_labeled.csv"), index=False)
    
    print("Processamento concluído com sucesso!")

if __name__ == "__main__":
    run_preprocessing()
