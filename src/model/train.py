import sys
import os
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Get the absolute path to the project root (the folder containing 'src')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add it to the sys.path so Python can find 'src'
if project_root not in sys.path:
    sys.path.append(project_root)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

from src.model.architecture import build_lstm_autoencoder
from src.data.preprocess import create_sequences


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def compute_reconstruction_errors(model, X):
    """
    Calcula o MAE médio por amostra.
    """
    X_pred = model.predict(X, batch_size=128, verbose=0)
    errors = np.mean(np.abs(X - X_pred), axis=(1, 2))
    return errors

def get_season(month: int) -> str:
    if month in [12, 1, 2]:  return 'verao'
    if month in [3, 4, 5]:   return 'outono'
    if month in [6, 7, 8]:   return 'inverno'
    return 'primavera'

def run_training():
    config = load_config()
    
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    window = config['model']['timesteps']
    n_features = len(features)
    
    # Parâmetros de treino
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    lr = config['model']['learning_rate']
    
    # Caminhos para salvar
    model_path = config['model']['model_path']
    thresholds_path = config['model']['thresholds_path']
    
    print("Carregando bases processadas...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
    df_train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    df_val_labeled = pd.read_csv(os.path.join(processed_dir, "val_labeled.csv"))
    
    print("Gerando sequências temporais...")
    X_train = create_sequences(df_train, features, window=window)
    
    # Gerar sequências de validação (completo)
    X_val_all = create_sequences(df_val_labeled, features, window=window)
    # Timestamps correspondentes ao fim de cada janela de 72h
    val_timestamps = pd.to_datetime(df_val_labeled['timestamp'].values[window-1:])
    val_is_anomaly = df_val_labeled['is_anomaly'].values[window-1:]
    
    # Para o treinamento do autoencoder, usamos todo o conjunto de treino
    # (que por definição da divisão temporal contém apenas anos considerados normais: 2018-2022)
    # E para validação durante o treino, usamos apenas a parte NORMAL do conjunto de validação
    # para evitar vazamento do evento anômalo nas métricas de convergência
    normal_val_mask = (val_is_anomaly == 0)
    X_val_normal = X_val_all[normal_val_mask]
    
    print(f"Shapes das sequências - X_train: {X_train.shape}, X_val_normal: {X_val_normal.shape}")
    
    # 3. Construção do modelo
    print("Construindo o LSTM Autoencoder...")
    # Podemos passar a variante aqui (padrão classic, configurável)
    model = build_lstm_autoencoder(timesteps=window, n_features=n_features, variant="classic")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
    model.summary()
    
    # Definindo Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    # Treinamento
    print("Iniciando o treinamento...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_normal, X_val_normal),
        callbacks=callbacks,
        shuffle=True
    )
    
    print(f"Modelo salvo em {model_path}")
    
    # Recarregar o melhor modelo
    best_model = tf.keras.models.load_model(model_path)
    
    # 4. Cálculo de Thresholds (usando a validação NORMAL)
    print("Calculando erros de reconstrução na validação normal para calibração do threshold...")
    val_normal_errors = compute_reconstruction_errors(best_model, X_val_normal)
    
    # Threshold Global (Percentil 97 na validação normal)
    p97_global = float(np.percentile(val_normal_errors, config['anomaly']['global_percentile']))
    p95_global = float(np.percentile(val_normal_errors, 95))
    p99_global = float(np.percentile(val_normal_errors, 99))
    
    print(f"Threshold Global (p97): {p97_global:.6f}")
    print(f"p95: {p95_global:.6f} | p99: {p99_global:.6f}")
    
    # Thresholds Sazonais
    seasons = {
        'verao': [12, 1, 2],
        'outono': [3, 4, 5],
        'inverno': [6, 7, 8],
        'primavera': [9, 10, 11]
    }
    
    # Mapeando meses dos timestamps da validação normal para as estações correspondentes
    val_normal_months = val_timestamps[normal_val_mask].month
    
    seasonal_thresholds = {}
    for season, months in seasons.items():
        season_mask = np.isin(val_normal_months, months)
        if np.sum(season_mask) > 0:
            errors_in_season = val_normal_errors[season_mask]
            th = float(np.percentile(errors_in_season, config['anomaly']['seasonal_percentile']))
            seasonal_thresholds[season] = th
        else:
            # Caso não tenha dados, preenche com o global
            seasonal_thresholds[season] = p97_global
            
    # Empacotar e salvar os thresholds
    thresholds_dict = {
        'global_p95': p95_global,
        'global_p97': p97_global,
        'global_p99': p99_global,
        'seasonal': seasonal_thresholds
    }
    
    os.makedirs(os.path.dirname(thresholds_path), exist_ok=True)
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=4)
        
    print(f"Thresholds sazonais salvos em {thresholds_path}:")
    print(json.dumps(thresholds_dict, indent=4))

if __name__ == "__main__":
    run_training()
