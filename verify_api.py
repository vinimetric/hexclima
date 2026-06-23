import json
import time
import requests
import pandas as pd

def verify_endpoints():
    url_base = "http://127.0.0.1:8000"
    
    print("Aguardando 2 segundos para o servidor FastAPI inicializar...")
    time.sleep(2)
    
    # 1. Testar Rota Raiz
    try:
        res = requests.get(f"{url_base}/")
        print("\n--- ROTA RAIZ ---")
        print(f"Status Code: {res.status_code}")
        print(res.json())
    except Exception as e:
        print(f"Erro de conexão com o servidor FastAPI: {e}")
        return
        
    # 2. Testar Rota de Métricas
    try:
        res = requests.get(f"{url_base}/metrics?dataset=validation")
        print("\n--- METRICAS DE VALIDAÇAO ---")
        print(f"Status Code: {res.status_code}")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Erro ao testar métricas: {e}")
        
    # 3. Testar Rota de Drift
    try:
        # Criando payload para drift
        payload = {
            "records": [
                {
                    "timestamp": "2026-01-01 00:00:00",
                    "precipitacao": 0.0,
                    "nivel_rio": 130.0,
                    "velocidade_vento": 8.0,
                    "temperatura": 22.0,
                    "umidade": 65.0,
                    "pressao": 1014.0,
                    "vazao": 45.0
                }
            ]
        }
        res = requests.post(f"{url_base}/drift", json=payload)
        print("\n--- DETECÇAO DE DRIFT ---")
        print(f"Status Code: {res.status_code}")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Erro ao testar drift: {e}")
        
    # 4. Testar Rota de Predição (Requer 72 horas)
    try:
        df_test = pd.read_csv("data/processed/test_labeled.csv").head(72)
        records = []
        for _, row in df_test.iterrows():
            records.append({
                "timestamp": str(row['timestamp']),
                "precipitacao": float(row['precipitacao']),
                "nivel_rio": float(row['nivel_rio']),
                "velocidade_vento": float(row['velocidade_vento']),
                "temperatura": float(row['temperatura']),
                "umidade": float(row['umidade']),
                "pressao": float(row['pressao']),
                "vazao": float(row['vazao'])
            })
            
        payload = {"records": records}
        res = requests.post(f"{url_base}/predict", json=payload)
        print("\n--- PREDIÇAO (/predict) ---")
        print(f"Status Code: {res.status_code}")
        print("Previsão e classificação de anomalia:")
        print(json.dumps(res.json()["prediction"], indent=2))
        print("Atribuição do erro por feature (Explicabilidade):")
        print(json.dumps(res.json()["explanation"][:3], indent=2)) # Primeiras 3 features
    except Exception as e:
        print(f"Erro ao testar predição: {e}")

if __name__ == "__main__":
    verify_endpoints()
