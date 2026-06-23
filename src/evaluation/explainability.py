import numpy as np
import pandas as pd

def explain_anomaly(model, window: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Usa a atribuição do erro de reconstrução por feature para identificar qual variável
    meteorológica ou hidrológica está mais anômala.
    
    Args:
        model: modelo Keras (LSTM Autoencoder)
        window: array de formato (timesteps, n_features)
        feature_names: lista de strings com os nomes das variáveis correspondentes
        
    Returns:
        df_explanation: pd.DataFrame contendo as features ordenadas pelo erro de reconstrução
    """
    # Ajusta o shape para o predict (adiciona dimensão do batch)
    if len(window.shape) == 2:
        window_batch = window[np.newaxis, ...]
    else:
        window_batch = window
        window = window[0] # Pega a primeira amostra do lote para explicação
        
    window_pred = model.predict(window_batch, verbose=0)[0]
    
    # MAE médio ao longo do tempo (eixos da janela temporal) para cada feature
    # window shape: (timesteps, n_features), window_pred shape: (timesteps, n_features)
    error_per_feature = np.mean(np.abs(window - window_pred), axis=0)
    
    # Calcular a contribuição percentual de cada feature para o erro total
    total_error = np.sum(error_per_feature)
    pct_contribution = (error_per_feature / total_error) * 100 if total_error > 0 else np.zeros_like(error_per_feature)
    
    df_explanation = pd.DataFrame({
        'feature': feature_names,
        'reconstruction_error': np.round(error_per_feature, 6),
        'contribution_pct': np.round(pct_contribution, 2)
    }).sort_values('reconstruction_error', ascending=False)
    
    return df_explanation

def explain_anomaly_shap(model, background_data: np.ndarray, test_instance: np.ndarray, feature_names: list):
    """
    Explica a anomalia usando SHAP (opcional, requer biblioteca 'shap').
    Usa um KernelExplainer aproximado devido à natureza sequencial do LSTM.
    """
    try:
        import shap
        
        # Como o Autoencoder reconstrói a sequência completa, explicamos a reconstrução média
        # Definimos uma função wrapper que recebe a entrada plana e retorna o erro médio
        def model_wrapper(x_flat):
            # x_flat shape: (batch_size, timesteps * n_features)
            n_samples = x_flat.shape[0]
            timesteps = test_instance.shape[0]
            n_feats = len(feature_names)
            
            x_reshaped = x_flat.reshape(n_samples, timesteps, n_feats)
            preds = model.predict(x_reshaped, verbose=0)
            # Retorna o MAE por amostra
            errors = np.mean(np.abs(x_reshaped - preds), axis=(1, 2))
            return errors
            
        # Aplanar os dados para o SHAP KernelExplainer
        bg_flat = background_data.reshape(background_data.shape[0], -1)
        inst_flat = test_instance.reshape(1, -1)
        
        # Usar um subconjunto de background para acelerar a computação
        explainer = shap.KernelExplainer(model_wrapper, bg_flat[:10])
        shap_values = explainer.shap_values(inst_flat, nsamples=50)
        
        # Remapear os valores SHAP para as variáveis (agregando por feature sobre o tempo)
        shap_reshaped = shap_values[0].reshape(test_instance.shape)
        shap_per_feature = np.mean(np.abs(shap_reshaped), axis=0)
        
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.round(shap_per_feature, 6)
        }).sort_values('shap_importance', ascending=False)
        
        return df_shap
    except ImportError:
        print("Biblioteca 'shap' não instalada ou não importável. Retornando None.")
        return None
