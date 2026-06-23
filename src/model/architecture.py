import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_autoencoder(timesteps: int = 72, n_features: int = 7, variant: str = "classic") -> Model:
    """
    LSTM Autoencoder para detecção de anomalias climáticas.
    
    Args:
        timesteps: tamanho da janela temporal (horas)
        n_features: número de variáveis meteorológicas/hidrológicas
        variant: variante da arquitetura ('classic', 'bidirectional', 'conv_lstm')
    
    Returns:
        model: modelo Keras compilado
    """
    inp = layers.Input(shape=(timesteps, n_features), name='input')

    if variant == "classic":
        # ── Encoder ──────────────────────────────────────────
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, name='enc_lstm_1')(inp)
        x = layers.LSTM(64, return_sequences=False, dropout=0.2, name='enc_lstm_2')(x)
        
        # ── Bottleneck ────────────────────────────────────────
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Decoder ──────────────────────────────────────────
        x = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x)
        out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)
        
    elif variant == "bidirectional":
        # ── Bidirectional Encoder ─────────────────────────────
        # Captura contexto passado e futuro
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2), name='enc_bilstm_1')(inp)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2), name='enc_bilstm_2')(x)
        
        # ── Bottleneck ────────────────────────────────────────
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Decoder ──────────────────────────────────────────
        x = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x)
        out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)
        
    elif variant == "conv_lstm":
        # ── Conv1D + LSTM Encoder ─────────────────────────────
        # Conv1D extrai padrões de correlação local de curto prazo
        x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='enc_conv1d_1')(inp)
        x = layers.MaxPool1D(pool_size=2, name='enc_pool_1')(x) # Reduz temporalidade para 36
        x = layers.LSTM(64, return_sequences=False, dropout=0.2, name='enc_lstm')(x)
        
        # ── Bottleneck ────────────────────────────────────────
        bottleneck = layers.Dense(32, activation='tanh', name='bottleneck')(x)
        
        # ── Decoder ──────────────────────────────────────────
        x = layers.RepeatVector(timesteps, name='repeat')(bottleneck)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2, name='dec_lstm_1')(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, name='dec_lstm_2')(x)
        out = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)
        
    else:
        raise ValueError(f"Variante '{variant}' desconhecida. Use 'classic', 'bidirectional', ou 'conv_lstm'.")

    model = Model(inp, out, name=f'hexclima_{variant}')
    return model

if __name__ == "__main__":
    # Testar construção das variantes
    for var in ["classic", "bidirectional", "conv_lstm"]:
        m = build_lstm_autoencoder(72, 7, variant=var)
        print(f"Modelo variante '{var}': {m.name} criado com sucesso. Total de parâmetros: {m.count_params()}")
