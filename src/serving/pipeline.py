import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from src.model.anomaly import detect_anomaly
from src.serving.alerts import dispatch_alert
from src.data.preprocess import handle_missing

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def inference_pipeline(latest_data: pd.DataFrame,
                       model,
                       scaler,
                       thresholds: dict) -> dict:
    """
    Orquestra o pipeline de inferência horária.
    Ingere as últimas 72 horas de dados brutos e retorna o diagnóstico.
    
    Args:
        latest_data: pd.DataFrame contendo pelo menos as últimas 72 horas de registros.
        model: modelo LSTM Autoencoder carregado.
        scaler: RobustScaler carregado.
        thresholds: dict de thresholds carregado.
        
    Returns:
        alert: dict com os resultados da detecção.
    """
    config = load_config()
    features = config['data']['features']
    timesteps = config['model']['timesteps']

    if len(latest_data) < timesteps:
        raise ValueError(f"Dados insuficientes para inferência. Necessário pelo menos {timesteps} horas. Recebido {len(latest_data)} horas.")

    # 1. Engenharia de features: calcular acumulados/médias móveis a partir do payload bruto
    #    O payload da API carrega as 7 features brutas; as 2 features de engenharia
    #    são calculadas aqui para que o scaler encontre todas as 9 colunas esperadas.
    df = latest_data.copy()
    if 'precipitacao' in df.columns and 'precip_24h' not in df.columns:
        df['precip_24h'] = df['precipitacao'].rolling(window=24, min_periods=1).sum()
    if 'nivel_rio' in df.columns and 'nivel_rio_ma_48h' not in df.columns:
        df['nivel_rio_ma_48h'] = df['nivel_rio'].rolling(window=48, min_periods=1).mean()

    # 2. Seleciona as últimas 72 horas com todas as 9 features
    window_raw = df.iloc[-timesteps:][features].copy()

    # 2b. Imputação de NaNs para garantir estabilidade da inferência
    window_raw = handle_missing(window_raw)

    # 2. Normaliza a janela usando o scaler
    window_scaled = scaler.transform(window_raw)
    
    # 3. Determina o timestamp da última leitura (momento atual)
    # Se o timestamp estiver no index ou nas colunas
    if 'timestamp' in latest_data.columns:
        last_timestamp = pd.to_datetime(latest_data.iloc[-1]['timestamp'])
    else:
        last_timestamp = pd.Timestamp(datetime.utcnow())
        
    # 4. Inferência e Detecção de Anomalia
    alert = detect_anomaly(model, window_scaled, thresholds, last_timestamp, scaler=scaler, feature_names=features)
    
    # 5. Ação baseada no nível do alerta
    if alert['level'] in ['ATENÇÃO', 'ALERTA', 'EMERGÊNCIA']:
        dispatch_alert(alert)
        
    return alert
