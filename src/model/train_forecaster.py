import os
import sys
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Anchor the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

from src.model.forecaster_architecture import build_realistic_forecaster

def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_forecasting_sequences(data: np.ndarray, timesteps: int, future_steps: int, stride: int = 1):
    """
    Cria os pares (X, y) para treinar o Forecaster.
    X: (N, timesteps, n_features)
    y: (N, future_steps, n_features)
    """
    X, y = [], []
    for i in range(0, len(data) - timesteps - future_steps + 1, stride):
        X.append(data[i : i + timesteps])
        y.append(data[i + timesteps : i + timesteps + future_steps])
    return np.array(X), np.array(y)

def train_forecaster():
    config = load_config()
    
    # Paths
    processed_dir = os.path.join(PROJECT_ROOT, config['data']['processed_dir'])
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    train_full_path = os.path.join(processed_dir, "train_full.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    
    features = config['data']['features']
    timesteps = config['model']['timesteps']
    future_steps = config['model'].get('future_steps', 12)  # Prever as próximas 12 horas por padrao
    
    print(f"Carregando dados de treino completos: {train_full_path}")
    df_train = pd.read_csv(train_full_path)
    df_val = pd.read_csv(val_path)
    
    data_train = df_train[features].values
    data_val = df_val[features].values
    
    print("Criando pares de sequencias (X, y) para Forecasting...")
    X_train, y_train = create_forecasting_sequences(data_train, timesteps, future_steps)
    X_val, y_val = create_forecasting_sequences(data_val, timesteps, future_steps)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    
    # Build Model
    model = build_realistic_forecaster(timesteps, len(features), future_steps)
    model.summary()
    
    # Callbacks
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "forecaster_rs_v1.h5")
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    batch_size = config['model']['batch_size']
    epochs = config['model']['epochs']
    
    print("Iniciando treinamento do Forecaster Realista...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"Treinamento concluído. Melhor modelo salvo em: {model_path}")

if __name__ == "__main__":
    train_forecaster()
