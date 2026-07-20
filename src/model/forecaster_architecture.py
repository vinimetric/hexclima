import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

def build_realistic_forecaster(timesteps: int, n_features: int, future_steps: int = 12) -> Model:
    """
    Constrói um modelo Sequence-to-Sequence para previsão multivariada.
    
    Args:
        timesteps: Tamanho da janela de histórico (ex: 72 horas)
        n_features: Número de variáveis preditoras (ex: 9)
        future_steps: Horizonte de previsão no futuro (ex: 12 horas)
        
    Returns:
        Modelo Keras compilado para Forecasting.
    """
    inputs = Input(shape=(timesteps, n_features), name="forecaster_input")
    
    # Encoder
    # O Encoder processa o histórico e extrai o contexto (estados finais)
    encoder_lstm, state_h, state_c = LSTM(64, return_state=True, name="forecaster_encoder")(inputs)
    encoder_states = [state_h, state_c]
    
    # Repeat Vector cria o molde para a sequência futura
    decoder_inputs = RepeatVector(future_steps, name="forecaster_repeat")(encoder_lstm)
    
    # Decoder
    # O Decoder usa os estados do Encoder para prever o futuro passo a passo
    decoder_lstm = LSTM(64, return_sequences=True, name="forecaster_decoder")(
        decoder_inputs, initial_state=encoder_states
    )
    
    # Camada de Saída
    outputs = TimeDistributed(Dense(n_features), name="forecaster_output")(decoder_lstm)
    
    model = Model(inputs=inputs, outputs=outputs, name="realistic_forecaster")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model
