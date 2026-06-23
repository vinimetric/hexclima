import os
import numpy as np
import pandas as pd
import tensorflow as tf

def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """
    Calcula o erro de reconstrução médio (MAE) por amostra.
    
    Args:
        model: Modelo Keras
        X: array de shape (N, timesteps, n_features) ou (timesteps, n_features)
    
    Returns:
        errors: array de shape (N,) contendo MAE de cada amostra
    """
    # Se for uma única amostra com shape (timesteps, n_features), adiciona dimensão de batch
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]
        
    X_pred = model.predict(X, verbose=0)
    # MAE médio sobre as dimensões de tempo e de features por amostra
    errors = np.mean(np.abs(X - X_pred), axis=(1, 2))
    return errors

def get_season(month: int) -> str:
    """
    Retorna a estação do ano correspondente ao mês.
    """
    if month in [12, 1, 2]:  return 'verao'
    if month in [3, 4, 5]:   return 'outono'
    if month in [6, 7, 8]:   return 'inverno'
    return 'primavera'

def detect_anomaly(model, window: np.ndarray, thresholds: dict, timestamp: pd.Timestamp) -> dict:
    """
    Calcula o score de anomalia e retorna o nível de alerta e informações da severidade.
    
    Args:
        model: modelo LSTM Autoencoder carregado
        window: array com formato (timesteps, n_features) ou (1, timesteps, n_features)
        thresholds: dicionário contendo os thresholds ('global_p97', 'seasonal', etc.)
        timestamp: objeto pd.Timestamp da leitura atual
    
    Returns:
        alert_info: dict com dados detalhados da inferência e criticidade
    """
    # Limpa shape se tiver dimensão de batch excedente
    if len(window.shape) == 3:
        if window.shape[0] == 1:
            window_to_predict = window[0]
        else:
            raise ValueError("O método detect_anomaly recebe apenas uma janela (1 amostra) por vez.")
    else:
        window_to_predict = window

    error = reconstruction_error(model, window_to_predict)[0]
    
    # Determinar a estação
    season = get_season(timestamp.month)
    
    # Obter o threshold correto (sazonal ou fallbacks)
    seasonal_thresholds = thresholds.get('seasonal', {})
    threshold = seasonal_thresholds.get(season, thresholds.get('global_p97', 0.0))
    
    # Severidade: percentual acima do threshold
    severity = (error - threshold) / threshold if threshold > 0 else 0.0

    # Classificação de níveis alinhados com a Defesa Civil do RS
    if error < threshold:
        level = 'NORMAL'
    elif severity < 0.2:
        # Erro até 20% acima do threshold
        level = 'ATENÇÃO'
    elif severity < 0.6:
        # Erro entre 20% e 60% acima do threshold
        level = 'ALERTA'
    else:
        # Erro acima de 60% do threshold
        level = 'EMERGÊNCIA'

    return {
        'timestamp': timestamp,
        'reconstruction_error': float(round(error, 6)),
        'threshold': float(round(threshold, 6)),
        'severity': float(round(severity, 4)),
        'level': level,
        'is_anomaly': bool(error > threshold),
        'season': season
    }
