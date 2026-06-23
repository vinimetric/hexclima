import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

def evaluate_anomaly_detector(y_true: np.ndarray,
                               scores: np.ndarray,
                               thresholds) -> dict:
    """
    Avalia o detector de anomalias com métricas balanceadas e específicas para risco de desastre.
    Para enchentes: recall >> precision (custo de falso negativo é muito alto).
    
    Args:
        y_true: array contendo rótulos reais (0 para normal, 1 para anomalia)
        scores: array contendo erro de reconstrução obtido pelo modelo
        thresholds: float ou array do mesmo tamanho de scores com os limites sazonais
        
    Returns:
        metrics: dicionário com Precision, Recall, F1, F2-score, AUROC, AUPRC e Matriz de Confusão.
    """
    # Se thresholds for um float/int simples, gera array broadcasted
    if isinstance(thresholds, (int, float, np.float32, np.float64)):
        threshold_array = np.full_like(scores, thresholds)
    else:
        threshold_array = np.array(thresholds)
        
    y_pred = (scores > threshold_array).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    
    # F2-score: penaliza falsos negativos (recall vale 2x mais que precision)
    beta = 2
    f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-9)
    
    # AUROC e AUPRC são independentes do threshold
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = 0.5  # Caso em que há apenas uma classe ativa no subconjunto
        
    try:
        auprc = average_precision_score(y_true, scores)
    except Exception:
        auprc = 0.0
        
    cm = confusion_matrix(y_true, y_pred)
    
    # False Alarm Rate (Taxa de Alarme Falso) = FP / (FP + TN)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'precision': float(round(precision, 4)),
        'recall':    float(round(recall, 4)),
        'f1':        float(round(f1, 4)),
        'f2':        float(round(f2, 4)),
        'far':       float(round(far, 4)),
        'auroc':     float(round(auroc, 4)),
        'auprc':     float(round(auprc, 4)),
        'confusion_matrix': cm.tolist()
    }

def compute_lead_time(anomaly_timestamps: list,
                      event_timestamps: list,
                      tolerance_hours: int = 72) -> dict:
    """
    Calcula quantas horas antes do pico de cheia o modelo dispara o primeiro alerta.
    Mínimo operacional: 6 horas (tempo padrão de resposta rápida da Defesa Civil RS).
    
    Args:
        anomaly_timestamps: lista ou série de objetos datetime/Timestamp contendo os alertas disparados
        event_timestamps: lista ou série de objetos datetime/Timestamp representando os picos das enchentes
        tolerance_hours: janela máxima retroativa para conectar um alerta a um evento
        
    Returns:
        lead_time_metrics: dict com média, mediana, mínimo de lead time e contadores de acerto/erro
    """
    lead_times = []
    
    # Converter para pd.Timestamp para facilitar operações de tempo
    anomaly_ts = [pd.Timestamp(t) for t in anomaly_timestamps]
    event_ts_list = [pd.Timestamp(t) for t in event_timestamps]
    
    for event_ts in event_ts_list:
        # Alertas que antecedem o evento e estão dentro do limite de tolerância (em horas)
        preceding = [
            t for t in anomaly_ts
            if 0 < (event_ts - t).total_seconds() / 3600 <= tolerance_hours
        ]
        if preceding:
            # Primeiro alerta (mais antigo dentro da janela)
            first_alert = min(preceding)
            lead_hours = (event_ts - first_alert).total_seconds() / 3600
            lead_times.append(lead_hours)
            
    if len(lead_times) > 0:
        mean_lead = np.mean(lead_times)
        median_lead = np.median(lead_times)
        min_lead = np.min(lead_times)
    else:
        mean_lead = 0.0
        median_lead = 0.0
        min_lead = 0.0
        
    detected = len(lead_times)
    missed = len(event_ts_list) - detected
    
    return {
        'mean_lead_hours':   float(round(mean_lead, 1)),
        'median_lead_hours': float(round(median_lead, 1)),
        'min_lead_hours':    float(round(min_lead, 1)),
        'events_detected':   int(detected),
        'events_missed':     int(missed)
    }
if __name__ == "__main__":
    # Teste rápido
    y_true = np.array([0, 0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.45, 0.2])
    print(evaluate_anomaly_detector(y_true, scores, 0.5))
