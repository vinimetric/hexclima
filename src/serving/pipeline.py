import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from src.model.anomaly import detect_anomaly
from src.serving.alerts import dispatch_alert

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
        
    # 1. Seleciona as últimas 72 horas de features
    window_raw = latest_data.iloc[-timesteps:][features].copy()
    
    # 2. Normaliza a janela usando o scaler
    window_scaled = scaler.transform(window_raw)
    
    # 3. Determina o timestamp da última leitura (momento atual)
    # Se o timestamp estiver no index ou nas colunas
    if 'timestamp' in latest_data.columns:
        last_timestamp = pd.to_datetime(latest_data.iloc[-1]['timestamp'])
    else:
        last_timestamp = pd.Timestamp(datetime.utcnow())
        
    # 4. Inferência e Detecção de Anomalia
    alert = detect_anomaly(model, window_scaled, thresholds, last_timestamp)
    
    # 5. Ação baseada no nível do alerta
    if alert['level'] in ['ATENÇÃO', 'ALERTA', 'EMERGÊNCIA']:
        dispatch_alert(alert)
        
    return alert
