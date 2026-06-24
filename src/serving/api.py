import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from scipy.stats import entropy

from src.serving.pipeline import inference_pipeline
from src.evaluation.explainability import explain_anomaly
from src.evaluation.metrics import evaluate_anomaly_detector, compute_lead_time

# Configurações do workspace
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

app = FastAPI(
    title="HexClima — API de Detecção de Anomalias Climáticas",
    description="API FastAPI do HexClima: monitoramento em tempo real de condições meteorológicas e hidrológicas extremas no Rio Grande do Sul via LSTM Autoencoder.",
    version="1.0.0"
)

# Variáveis globais carregadas na inicialização
model = None
scaler = None
thresholds = None
config = load_config()

class MeteorologicalRecord(BaseModel):
    timestamp: str
    precipitacao: float
    nivel_rio: float
    velocidade_vento: float
    temperatura: float
    umidade: float
    pressao: float
    vazao: float

class InferencePayload(BaseModel):
    records: List[MeteorologicalRecord]

class DriftPayload(BaseModel):
    records: List[MeteorologicalRecord]

@app.on_event("startup")
def startup_event():
    global model, scaler, thresholds
    model_path = config['model']['model_path']
    scaler_path = config['model']['scaler_path']
    thresholds_path = config['model']['thresholds_path']
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()})
            print(f"Modelo carregado de {model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
    else:
        print(f"AVISO: Modelo não encontrado em {model_path}. Execute o treinamento primeiro.")
        
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler carregado de {scaler_path}")
    else:
        print(f"AVISO: Scaler não encontrado em {scaler_path}.")
        
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
        print(f"Thresholds carregados de {thresholds_path}")
    else:
        print(f"AVISO: Thresholds não encontrados em {thresholds_path}.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "thresholds_loaded": thresholds is not None,
        "api_name": "LSTM Autoencoder RS Anomaly Detection API"
    }

@app.post("/predict")
def predict(payload: InferencePayload):
    global model, scaler, thresholds
    if model is None or scaler is None or thresholds is None:
        raise HTTPException(
            status_code=503, 
            detail="Serviço indisponível. O modelo, o scaler ou os thresholds ainda não foram gerados/carregados."
        )
        
    timesteps = config['model']['timesteps']
    if len(payload.records) < timesteps:
        raise HTTPException(
            status_code=400, 
            detail=f"Registros insuficientes. O modelo requer exatamente {timesteps} horas consecutivas de histórico para inferência. Fornecido: {len(payload.records)}"
        )
        
    # Converter payload para DataFrame
    data_list = [r.dict() for r in payload.records]
    df = pd.DataFrame(data_list)
    
    # Rodar o pipeline de inferência
    try:
        alert_info = inference_pipeline(df, model, scaler, thresholds)
        
        # Obter explicabilidade da janela
        features = config['data']['features']
        # Precisamos normalizar a janela para explicar a reconstrução correta
        window_raw = df.iloc[-timesteps:][features].copy()
        window_scaled = scaler.transform(window_raw)
        
        explanation_df = explain_anomaly(model, window_scaled, features)
        explanation = explanation_df.to_dict(orient='records')
        
        # Anexar explicação ao dicionário de resposta
        response = {
            "prediction": alert_info,
            "explanation": explanation
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no pipeline: {str(e)}")

@app.get("/metrics")
def get_metrics(dataset: str = "validation"):
    """
    Retorna métricas de validação ou teste calculadas com base nas séries temporais salvas.
    """
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    window = config['model']['timesteps']
    thresholds_path = config['model']['thresholds_path']
    
    if dataset == "validation":
        file_path = os.path.join(processed_dir, "val_labeled.csv")
    elif dataset == "test":
        file_path = os.path.join(processed_dir, "test_labeled.csv")
    else:
        raise HTTPException(status_code=400, detail="Parâmetro 'dataset' inválido. Use 'validation' ou 'test'.")
        
    if not os.path.exists(file_path) or model is None or not os.path.exists(thresholds_path):
        raise HTTPException(status_code=404, detail="Dados rotulados ou modelo/thresholds não disponíveis para cálculo.")
        
    df = pd.read_csv(file_path)
    
    # Carregar os thresholds
    with open(thresholds_path, 'r') as f:
        th_dict = json.load(f)
        
    # Gerar sequências e scores
    data_scaled = df[features].values
    X = []
    for i in range(0, len(data_scaled) - window + 1):
        X.append(data_scaled[i:i + window])
    X = np.array(X)
    
    # Scores
    X_pred = model.predict(X, batch_size=128, verbose=0)
    scores = np.mean(np.abs(X - X_pred), axis=(1, 2))
    
    y_true = df['is_anomaly'].values[window-1:]
    timestamps = pd.to_datetime(df['timestamp'].values[window-1:])
    
    # Calcular thresholds dinâmicos por estação correspondente a cada janela
    seasonal_thresholds = th_dict.get('seasonal', {})
    default_threshold = th_dict.get('global_p97', 0.0)
    
    thresholds_array = []
    for ts in timestamps:
        month = ts.month
        if month in [12, 1, 2]:    season = 'verao'
        elif month in [3, 4, 5]:  season = 'outono'
        elif month in [6, 7, 8]:  season = 'inverno'
        else:                     season = 'primavera'
        
        thresholds_array.append(seasonal_thresholds.get(season, default_threshold))
        
    thresholds_array = np.array(thresholds_array)
    
    # Calcular métricas principais
    eval_metrics = evaluate_anomaly_detector(y_true, scores, thresholds_array)
    
    # Calcular Lead Time operacional
    # Identificar timestamps de alertas
    alerts_mask = scores > thresholds_array
    alert_timestamps = timestamps[alerts_mask].tolist()
    
    # Identificar picos reais de eventos
    # Enchente Set/2023 peak: ~ 2023-09-05
    # Enchente Mai/2024 peak: ~ 2024-05-04
    event_peaks = []
    if dataset == "validation":
        event_peaks = [pd.Timestamp("2023-09-05 12:00:00")]
    else:
        event_peaks = [pd.Timestamp("2024-05-04 12:00:00")]
        
    lead_time_metrics = compute_lead_time(alert_timestamps, event_peaks, tolerance_hours=72)
    
    return {
        "dataset": dataset,
        "classification_metrics": eval_metrics,
        "operational_metrics": lead_time_metrics
    }

@app.post("/drift")
def check_drift(payload: DriftPayload):
    """
    Monitora drift nas features usando Divergência KL contra o conjunto de treino.
    """
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    threshold_kl = config['anomaly'].get('drift_threshold_kl', 0.1)
    
    train_path = os.path.join(processed_dir, "train.csv")
    if not os.path.exists(train_path):
        raise HTTPException(status_code=404, detail="Dados de treino de referência não encontrados.")
        
    df_train = pd.read_csv(train_path)
    
    # Converter dados novos
    data_list = [r.dict() for r in payload.records]
    df_current = pd.DataFrame(data_list)
    
    # Aplicar o RobustScaler nas duas tabelas se disponível
    global scaler
    if scaler is not None:
        df_train_scaled = scaler.transform(df_train[features])
        df_current_scaled = scaler.transform(df_current[features])
    else:
        df_train_scaled = df_train[features].values
        df_current_scaled = df_current[features].values
        
    drift_scores = {}
    for i, feature in enumerate(features):
        # Gerar histogramas emparelhados
        hist_ref, bins = np.histogram(df_train_scaled[:, i], bins=50, density=True)
        hist_cur, _    = np.histogram(df_current_scaled[:, i], bins=bins, density=True)

        # Adiciona epsilon para evitar log(0)
        hist_ref = hist_ref + 1e-10
        hist_cur = hist_cur + 1e-10

        kl_div = entropy(hist_ref, hist_cur)
        drift_scores[feature] = {
            'kl_divergence': float(round(kl_div, 4)),
            'drift_detected': bool(kl_div > threshold_kl)
        }
        
    total_drifts = sum([1 for f in drift_scores.values() if f['drift_detected']])
    
    return {
        "drift_scores": drift_scores,
        "total_features_with_drift": total_drifts,
        "recommend_retraining": total_drifts >= 2
    }
