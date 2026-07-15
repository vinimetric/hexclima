import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_autoencoder(timesteps: int = 72, n_features: int = 7, variant: str = "classic", future_steps: int = 0) -> Model:
    """
    LSTM Autoencoder para detecção de anomalias climáticas e forecasting.
    
    Args:
        timesteps: tamanho da janela temporal (horas)
        n_features: número de variáveis meteorológicas/hidrológicas
        variant: variante da arquitetura ('classic', 'bidirectional', 'conv_lstm')
        future_steps: número de horas futuras a prever (se > 0, adiciona branch de forecasting)
    
    Returns:
        model: modelo Keras compilado
    """
    inp = layers.Input(shape=(timesteps, n_features), name='input')

    if variant == "classic":
        # ── Encoder (Shared) ──────────────────────────────────
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, name='enc_lstm_1')(inp)
        x = layers.LSTM(64, return_sequences=False, dropout=0.2, name='enc_lstm_2')(x)
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Reconstruction Decoder ───────────────────────────
        x_rec = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x_rec = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x_rec)
        x_rec = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x_rec)
        
        if future_steps > 0:
            out_rec = layers.TimeDistributed(layers.Dense(n_features), name='output_recon')(x_rec)
            # ── Forecasting Decoder ────────────────────────────
            x_fore = layers.RepeatVector(future_steps, name='repeat_fore')(bottleneck)
            x_fore = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_fore_lstm_1')(x_fore)
            x_fore = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_fore_lstm_2')(x_fore)
            out_fore = layers.TimeDistributed(layers.Dense(n_features), name='output_fore')(x_fore)
            out = [out_rec, out_fore]
        else:
            out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x_rec)
        
    elif variant == "bidirectional":
        # ── Bidirectional Encoder (Shared) ───────────────────
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2), name='enc_bilstm_1')(inp)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2), name='enc_bilstm_2')(x)
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Reconstruction Decoder ───────────────────────────
        x_rec = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x_rec = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x_rec)
        x_rec = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x_rec)
        
        if future_steps > 0:
            out_rec = layers.TimeDistributed(layers.Dense(n_features), name='output_recon')(x_rec)
            # ── Forecasting Decoder ────────────────────────────
            x_fore = layers.RepeatVector(future_steps, name='repeat_fore')(bottleneck)
            x_fore = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_fore_lstm_1')(x_fore)
            x_fore = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_fore_lstm_2')(x_fore)
            out_fore = layers.TimeDistributed(layers.Dense(n_features), name='output_fore')(x_fore)
            out = [out_rec, out_fore]
        else:
            out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x_rec)
        
    elif variant == "conv_lstm":
        # ── Conv1D + LSTM Encoder (Shared) ───────────────────
        x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='enc_conv1d_1')(inp)
        x = layers.MaxPool1D(pool_size=2, name='enc_pool_1')(x) 
        x = layers.LSTM(64, return_sequences=False, dropout=0.2, name='enc_lstm')(x)
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Reconstruction Decoder ───────────────────────────
        x_rec = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x_rec = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x_rec)
        x_rec = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x_rec)
        
        if future_steps > 0:
            out_rec = layers.TimeDistributed(layers.Dense(n_features), name='output_recon')(x_rec)
            # ── Forecasting Decoder ────────────────────────────
            x_fore = layers.RepeatVector(future_steps, name='repeat_fore')(bottleneck)
            x_fore = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_fore_lstm_1')(x_fore)
            x_fore = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_fore_lstm_2')(x_fore)
            out_fore = layers.TimeDistributed(layers.Dense(n_features), name='output_fore')(x_fore)
            out = [out_rec, out_fore]
        else:
            out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x_rec)
        
    else:
        raise ValueError(f"Variante '{variant}' desconhecida. Use 'classic', 'bidirectional', ou 'conv_lstm'.")

    model_name = f'hexclima_joint_{variant}' if future_steps > 0 else f'hexclima_{variant}'
    model = Model(inp, out, name=model_name)
    return model

if __name__ == "__main__":
    # Testar construção das variantes
    for var in ["classic", "bidirectional", "conv_lstm"]:
        m = build_lstm_autoencoder(72, 7, variant=var)
        print(f"Modelo variante '{var}': {m.name} criado com sucesso. Total de parâmetros: {m.count_params()}")
