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
    if isinstance(X_pred, list):
        X_pred = X_pred[0] # Get reconstruction output
        
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

def detect_anomaly(model, window: np.ndarray, thresholds: dict, timestamp: pd.Timestamp, scaler=None, feature_names=None) -> dict:
    """
    Calcula o score de anomalia e retorna o nível de alerta e informações da severidade,
    incluindo previsões futuras e alertas futuros se o modelo suportar forecasting.
    
    Args:
        model: modelo LSTM Autoencoder carregado
        window: array com formato (timesteps, n_features) ou (1, timesteps, n_features)
        thresholds: dicionário contendo os thresholds ('global_p97', 'seasonal', etc.)
        timestamp: objeto pd.Timestamp da leitura atual
        scaler: RobustScaler (opcional, para decodificar predições futuras)
        feature_names: lista de features (opcional, para decodificar predições futuras)
    
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

    # Fazer a predição. Pode retornar apenas reconstrução ou [reconstrução, forecast]
    preds = model.predict(window_to_predict[np.newaxis, ...], verbose=0)
    if isinstance(preds, list):
        recon_pred = preds[0][0]
        fore_pred = preds[1][0] # shape: (future_steps, n_features)
    else:
        recon_pred = preds[0]
        fore_pred = None

    error = np.mean(np.abs(window_to_predict - recon_pred))
    
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

    alert_info = {
        'timestamp': timestamp,
        'reconstruction_error': float(round(error, 6)),
        'threshold': float(round(threshold, 6)),
        'severity': float(round(severity, 4)),
        'level': level,
        'is_anomaly': bool(error > threshold),
        'season': season
    }

    # Se tiver forecasting, decodificar e calcular alerta futuro
    if fore_pred is not None and scaler is not None and feature_names is not None:
        # 1. Decodificar previsões futuras de volta para unidades originais
        fore_df = pd.DataFrame(fore_pred, columns=feature_names)
        fore_decoded = scaler.inverse_transform(fore_df)
        
        forecast_records = []
        for t_idx in range(len(fore_decoded)):
            future_ts = timestamp + pd.Timedelta(hours=t_idx + 1)
            record = {
                'timestamp': str(future_ts)
            }
            for f_idx, f_name in enumerate(feature_names):
                record[f_name] = float(round(fore_decoded[t_idx, f_idx], 2))
            forecast_records.append(record)
            
        alert_info['forecast'] = forecast_records
        
        # 2. Avaliar anomalias futuras na janela mesclada (fim do passado + previsões futuras)
        future_steps = fore_pred.shape[0]
        # Pega as últimas (72 - future_steps) horas da janela atual
        past_part = window_to_predict[future_steps:] 
        # Junta com as previsões futuras (ambos normalizados)
        future_window = np.vstack([past_part, fore_pred])
        
        # Reconstruir a janela futura estimada
        future_preds = model.predict(future_window[np.newaxis, ...], verbose=0)
        future_recon_pred = future_preds[0][0] if isinstance(future_preds, list) else future_preds[0]
        future_error = np.mean(np.abs(future_window - future_recon_pred))
        
        future_target_ts = timestamp + pd.Timedelta(hours=future_steps)
        future_season = get_season(future_target_ts.month)
        future_threshold = seasonal_thresholds.get(future_season, thresholds.get('global_p97', 0.0))
        future_severity = (future_error - future_threshold) / future_threshold if future_threshold > 0 else 0.0
        
        if future_error < future_threshold:
            future_level = 'NORMAL'
        elif future_severity < 0.2:
            future_level = 'ATENÇÃO'
        elif future_severity < 0.6:
            future_level = 'ALERTA'
        else:
            future_level = 'EMERGÊNCIA'
            
        alert_info['future_anomaly'] = {
            'reconstruction_error': float(round(future_error, 6)),
            'threshold': float(round(future_threshold, 6)),
            'severity': float(round(future_severity, 4)),
            'level': future_level,
            'is_anomaly': bool(future_error > future_threshold),
            'season': future_season,
            'lead_time_hours': future_steps
        }

    return alert_info
