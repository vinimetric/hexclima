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

def compute_reconstruction_errors_MAE(model, X):
    """
    Calcula o MAE medio por amostra.
    Suporta modelos simples e Joint (saida lista [recon, fore]).
    """
    X_pred = model.predict(X, batch_size=128, verbose=0)
    # Modelo Joint retorna lista [output_recon, output_fore] — pega so a reconstrucao
    if isinstance(X_pred, list):
        X_pred = X_pred[0]
    errors = np.mean(np.abs(X - X_pred), axis=(1, 2))
    return errors

def get_season(month: int) -> str:
    if month in [12, 1, 2]:  return 'verao'
    if month in [3, 4, 5]:   return 'outono'
    if month in [6, 7, 8]:   return 'inverno'
    return 'primavera'

def create_sequences_joint(df, features, window=72, future_steps=12, stride=1):
    """
    Cria sequências temporais de entrada e os alvos futuros para forecasting.
    """
    data = df[features].values
    X = []
    Y = []
    for i in range(0, len(data) - window - future_steps + 1, stride):
        X.append(data[i:i + window])
        Y.append(data[i + window:i + window + future_steps])
    return np.array(X), np.array(Y)

def run_training():
    config = load_config()
    
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    window = config['model']['timesteps']
    future_steps = config['model'].get('future_steps', 0)
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

    print(f"Gerando sequencias temporais (future_steps={future_steps})...")
    if future_steps > 0:
        X_train, Y_train_fore = create_sequences_joint(df_train, features, window=window, future_steps=future_steps)
        X_val_all, Y_val_all_fore = create_sequences_joint(df_val_labeled, features, window=window, future_steps=future_steps)
        # Slicing timestamps and labels to match the joint sequences
        val_timestamps = pd.to_datetime(df_val_labeled['timestamp'].values[window-1 : -future_steps])
        val_is_anomaly = df_val_labeled['is_anomaly'].values[window-1 : -future_steps]
    else:
        X_train = create_sequences(df_train, features, window=window)
        X_val_all = create_sequences(df_val_labeled, features, window=window)
        val_timestamps = pd.to_datetime(df_val_labeled['timestamp'].values[window-1:])
        val_is_anomaly = df_val_labeled['is_anomaly'].values[window-1:]

    print(f"X_train shape: {X_train.shape}")
    
    # Para o treinamento do autoencoder, usamos todo o conjunto de treino
    # E para validacao durante o treino, usamos apenas a parte NORMAL do conjunto de validacao
    normal_val_mask = (val_is_anomaly == 0)
    X_val_normal = X_val_all[normal_val_mask]
    
    if future_steps > 0:
        Y_val_normal_fore = Y_val_all_fore[normal_val_mask]
        print(f"Shapes das sequencias - X_train: {X_train.shape}, Y_train_fore: {Y_train_fore.shape}, X_val_normal: {X_val_normal.shape}")
    else:
        print(f"Shapes das sequencias - X_train: {X_train.shape}, X_val_normal: {X_val_normal.shape}")
    
    # 3. Construcao do modelo
    print("Construindo o LSTM Autoencoder...")
    model = build_lstm_autoencoder(timesteps=window, n_features=n_features, variant="classic", future_steps=future_steps)
    
    if future_steps > 0:
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss={'output_recon': 'mae', 'output_fore': 'mae'},
            loss_weights={'output_recon': 1.0, 'output_fore': 1.0}
        )
    else:
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
    if future_steps > 0:
        history = model.fit(
            X_train, {'output_recon': X_train, 'output_fore': Y_train_fore},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_normal, {'output_recon': X_val_normal, 'output_fore': Y_val_normal_fore}),
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )
    else:
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_normal, X_val_normal),
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )
    
    print(f"Modelo salvo em {model_path}")
    
    # Recarregar o melhor modelo (carrega sem compilar pois so faremos inferenca)
    best_model = tf.keras.models.load_model(model_path, compile=False)
    
    # 4. Calculo de Thresholds (usando a validacao NORMAL)
    print("Calculando erros de reconstrucao na validacao normal para calibracao do threshold...")
    val_normal_errors = compute_reconstruction_errors_MAE(best_model, X_val_normal)
    
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
