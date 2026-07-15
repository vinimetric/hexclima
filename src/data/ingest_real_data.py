import os
import yaml
import pandas as pd
import numpy as np
import glob

# Consistência de caminhos relativos ao arquivo
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def run_ingestion():
    config = load_config()
    raw_inmet_dir = config['data']['raw_inmet_dir']
    station_code = config['data']['station_code']
    raw_dir = config['data']['raw_dir']
    
    # 1. Encontrar o arquivo correspondente à estação configurada
    search_pattern = os.path.join(raw_inmet_dir, f"dados_{station_code}_H_*.csv")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para a estação {station_code} em {raw_inmet_dir}")
    
    inmet_file_path = matching_files[0]
    print(f"Lendo dados da estação {station_code} de: {inmet_file_path}")
    
    # 2. Carregar o arquivo pulando os metadados (primeiras 10 linhas)
    df_raw = pd.read_csv(inmet_file_path, skiprows=10, sep=';', encoding='latin1')
    
    # Remover colunas desnecessárias ou completamente vazias (como 'Unnamed: 22')
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')]
    
    # 3. Criar a coluna de timestamp
    print("Processando timestamps...")
    # 'Hora Medicao' está no formato HHMM (como int64: 0, 100, 1200, 2300)
    # Formata como string preenchendo com zeros à esquerda
    df_raw['hora_str'] = df_raw['Hora Medicao'].apply(lambda x: f"{x:04d}")
    df_raw['timestamp_str'] = df_raw['Data Medicao'] + ' ' + df_raw['hora_str']
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp_str'], format='%Y-%m-%d %H%M', errors='coerce')
    
    # Dropar NaNs nos timestamps e remover duplicatas
    df_raw = df_raw.dropna(subset=['timestamp'])
    df_raw = df_raw.drop_duplicates(subset=['timestamp'])
    
    # Mapeamento de colunas brutas do INMET para as variáveis do projeto
    column_mapping = {
        'PRECIPITACAO TOTAL, HORARIO(mm)': 'precipitacao',
        'VENTO, VELOCIDADE HORARIA(m/s)': 'velocidade_vento',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)': 'temperatura',
        'UMIDADE RELATIVA DO AR, HORARIA(%)': 'umidade',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)': 'pressao'
    }
    
    # Se existirem nomes com enconding quebrado, tratar (ex: Kj/mÂ²)
    # Vamos renomear com base nas substrings para evitar incompatibilidade de caracteres
    renamed_cols = {}
    for col in df_raw.columns:
        for raw_key, clean_key in column_mapping.items():
            if raw_key.split('(')[0] in col: # busca correspondência parcial
                renamed_cols[col] = clean_key
                break
                
    df_mapped = df_raw.rename(columns=renamed_cols)
    
    # Manter apenas as colunas mapeadas e o timestamp
    cols_to_keep = ['timestamp'] + list(column_mapping.values())
    df_mapped = df_mapped[[c for c in cols_to_keep if c in df_mapped.columns]]
    
    # 4. Criar um range horário completo para evitar lacunas na série temporal
    min_date = df_mapped['timestamp'].min()
    max_date = df_mapped['timestamp'].max()
    print(f"Dados originais vão de {min_date} até {max_date}")
    
    full_range = pd.date_range(start=min_date, end=max_date, freq='h')
    df_mapped = df_mapped.set_index('timestamp').reindex(full_range).reset_index().rename(columns={'index': 'timestamp'})
    
    # Converter velocidade do vento de m/s para km/h (* 3.6)
    if 'velocidade_vento' in df_mapped.columns:
        df_mapped['velocidade_vento'] = df_mapped['velocidade_vento'] * 3.6
        
    # Imputar os NaNs usando uma interpolação linear básica seguida por ffill/bfill
    # para garantir que tenhamos valores válidos para simular a hidrologia
    print("Tratando valores ausentes nos dados meteorológicos...")
    df_mapped = df_mapped.interpolate(method='linear', limit=12)
    try:
        df_mapped = df_mapped.ffill().bfill()
    except (TypeError, AttributeError):
        df_mapped = df_mapped.fillna(method='ffill').fillna(method='bfill')
        
    # Salvar inmet.csv
    os.makedirs(raw_dir, exist_ok=True)
    inmet_output = os.path.join(raw_dir, "inmet.csv")
    df_mapped.to_csv(inmet_output, index=False)
    print(f"Dados meteorológicos reais salvos em: {inmet_output}. Total de registros: {len(df_mapped)}")
    
    # 5. Simulação da hidrologia (vazão e nível do rio) correlacionada com a precipitação real
    print("Simulando dados hidrológicos (ANA/CPRM) baseados na precipitação real...")
    precip = df_mapped['precipitacao'].values
    n_hours = len(df_mapped)
    
    vazao = np.zeros(n_hours)
    base_vazao = 50.0  # m³/s vazão de base
    current_q = base_vazao
    
    # Filtro autorregressivo de resposta hidrológica (lag)
    alpha = 0.985  # Coeficiente de decaimento (bacia de resposta lenta)
    beta = 12.0    # Sensibilidade à chuva
    
    for t in range(n_hours):
        # A vazão acumula a chuva com decaimento exponencial
        current_q = alpha * current_q + (1 - alpha) * base_vazao + beta * precip[t]
        vazao[t] = current_q
        
    # Nível do rio (H) determinado pela curva de chaveamento clássica H = c * Q^d + ruído
    nivel_rio = 120.0 + 8.5 * np.power(vazao, 0.58) + np.random.normal(0.0, 3.0, n_hours)
    
    # Ajustar vazão com pequeno ruído e impor limites mínimos
    vazao = vazao + np.random.normal(0.0, 1.5, n_hours)
    vazao = np.clip(vazao, 5.0, None)
    nivel_rio = np.clip(nivel_rio, 20.0, None)
    
    df_ana = pd.DataFrame({
        'timestamp': df_mapped['timestamp'],
        'nivel_rio': np.round(nivel_rio, 1),
        'vazao': np.round(vazao, 1)
    })
    
    ana_output = os.path.join(raw_dir, "ana.csv")
    df_ana.to_csv(ana_output, index=False)
    print(f"Dados hidrológicos simulados salvos em: {ana_output}. Total de registros: {len(df_ana)}")
    print("Ingestão concluída com sucesso!")

if __name__ == "__main__":
    run_ingestion()
