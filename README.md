# HexClima 🌊🧠
### Detecção de Anomalias Climáticas no Rio Grande do Sul via LSTM Autoencoder

> **Paradigma:** Aprendizado de normalidade — o modelo é treinado **exclusivamente** em períodos climáticos normais e detecta eventos extremos (enchentes, temporais) pelo alto **erro de reconstrução** das séries temporais meteorológicas e hidrológicas.

---

## Sumário

1. [O que é o MemClimate](#o-que-é-o-memclimate)
2. [O que o modelo faz (e o que não faz)](#o-que-o-modelo-faz-e-o-que-não-faz)
3. [Arquitetura](#arquitetura)
4. [Fontes de Dados](#fontes-de-dados)
5. [Estrutura do Projeto](#estrutura-do-projeto)
6. [Instalação e Execução](#instalação-e-execução)
7. [API FastAPI](#api-fastapi)
8. [Níveis de Alerta](#níveis-de-alerta)
9. [Métricas de Avaliação](#métricas-de-avaliação)
10. [Docker](#docker)
11. [Roadmap](#roadmap)
12. [Referências](#referências)

---

## O que é o HexClima

O **HexClima** é um sistema de monitoramento climático em tempo real baseado em aprendizado de máquina para o Rio Grande do Sul. Ele utiliza um **LSTM Autoencoder** que aprende os padrões normais de comportamento climático e hidrológico ao longo de anos de dados históricos. Quando o padrão atual desvia significativamente do que o modelo aprendeu como "normal", um alerta é disparado.

O nome une **Hexa** (hexagonal, referenciando a estrutura em rede das bacias hidrográficas e a topologia dos dados) com **Clima** (domínio de aplicação) — o modelo mapeia a malha climática do RS como uma grade hexagonal de memória temporal.

### Evento-alvo de validação
O sistema foi validado contra a **Grande Enchente do RS de Maio de 2024** — o maior desastre climático da história do estado, afetando mais de 220 municípios.

---

## O que o modelo faz (e o que não faz)

### ✅ Capacidades atuais

| Capacidade | Detalhe |
|---|---|
| **Detecção de anomalia regional** | Identifica quando as condições das últimas 72h estão fora do padrão histórico |
| **Alerta antecipado implícito** | Detecta a deterioração gradual das condições **antes** do pico da enchente (lead time emergente) |
| **Classificação de severidade** | Quatro níveis: NORMAL → ATENÇÃO → ALERTA → EMERGÊNCIA |
| **Thresholds sazonais** | Limites adaptados por estação (verão chuvoso vs. inverno seco do RS) |
| **Explicabilidade por feature** | Identifica qual variável (chuva, nível do rio, pressão...) mais contribui para o alerta |
| **Detecção de data drift** | Monitora se a distribuição dos dados mudou e o modelo precisa ser retreinado |
| **API REST em tempo real** | Inferência via FastAPI com documentação interativa (Swagger) |

### ❌ Limitações atuais (fora do escopo v1)

| Limitação | Explicação |
|---|---|
| **Previsão futura explícita** | O modelo detecta o estado *presente*, não prevê "em 2h haverá anomalia" (requer arquitetura Seq2Seq) |
| **Mapa de risco por rua** | Sem componente espacial/GIS — não indica qual rua específica vai alagar |
| **Tempo até o alagamento** | Não estima "Rua X alaga em ~35 min" (requer dados geoespaciais + modelo supervisionado) |
| **Inferência hiperlocal** | Sem Modelo Digital de Terreno (MDT) integrado |

> Veja o [Roadmap](#roadmap) para as extensões planejadas que cobrem estas capacidades.

---

## Arquitetura

### Diagrama do LSTM Autoencoder

```
  Input            Encoder               Bottleneck       Decoder              Output
(72h, 7F)  →  LSTM(128) → LSTM(64)  →  Dense(32)  →  RepeatVec → LSTM(64) → LSTM(128) → Dense(7F)
                                           ↑
                                    Representação latente
                                    do padrão "normal"

  Erro de Reconstrução (MAE) = |Input - Output|
  Se erro > threshold sazonal → ANOMALIA
```

### Variantes disponíveis

| Variante | Arquivo | Quando usar |
|---|---|---|
| `classic` | `architecture.py` | Padrão — melhor custo/benefício |
| `bidirectional` | `architecture.py` | +10–15% precisão, mais lento |
| `conv_lstm` | `architecture.py` | Dados com padrões locais diários/semanais |

### Janelamento temporal

```
Dado bruto horário  →  Janelas deslizantes de 72h  →  Autoencoder
   (stride = 1h)              (N, 72, 7)
```

---

## Fontes de Dados

| Fonte | Variáveis | Tipo |
|---|---|---|
| **INMET** | Precipitação, Temperatura, Umidade, Pressão, Vento | Meteorológico |
| **ANA / CPRM** | Nível dos rios, Vazão fluvial | Hidrológico |
| **CEMADEN** | Alertas de desastres naturais | Eventos |
| **ERA5 (ECMWF)** | Reanálise climática global (1940–hoje) | Referência |
| **Dados sintéticos** | Gerados pelo `fetch_inmet.py` e `fetch_ana.py` com física realista | Fallback/Demo |

> **Nota:** Os scripts de ingestão incluem um gerador de dados sintéticos fisicamente consistentes como fallback, permitindo rodar o sistema completo sem acesso às APIs externas.

---

## Estrutura do Projeto

```
lstm_autoencoder_rs/          ← raiz do projeto (MemClimate)
│
├── config/
│   └── config.yaml           # Hiperparâmetros, features, thresholds e caminhos
│
├── data/
│   ├── raw/                  # Séries brutas (INMET / ANA ou sintéticas)
│   ├── processed/            # Dados normalizados, particionados e rotulados
│   └── events/               # Eventos históricos de enchente para validação
│
├── models/
│   ├── lstm_ae_rs_v1.h5      # Pesos do modelo Keras treinado
│   ├── scaler_rs_v1.pkl      # RobustScaler serializado (fit no treino)
│   └── thresholds_rs_v1.json # Percentis sazonais (p95, p97, p99)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploração e visualização dos dados brutos
│   ├── 02_preprocessing.ipynb# Tratamento de NaN, escalonamento e janelamento
│   ├── 03_training.ipynb     # Treinamento e calibração de thresholds
│   └── 04_evaluation.ipynb   # F2-score, Lead Time e Explicabilidade
│
├── src/
│   ├── data/
│   │   ├── fetch_inmet.py    # Ingestão/simulação de dados meteorológicos
│   │   ├── fetch_ana.py      # Ingestão/simulação de dados hidrológicos
│   │   └── preprocess.py     # Merge, NaN handling, RobustScaler, sliding windows
│   │
│   ├── model/
│   │   ├── architecture.py   # Definição das variantes do LSTM-AE (Keras)
│   │   ├── train.py          # Loop de treinamento + calibração sazonal de thresholds
│   │   └── anomaly.py        # Cálculo de erro de reconstrução e classificação de alerta
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # F2-score, Lead Time, AUROC, AUPRC, FAR
│   │   └── explainability.py # Atribuição do erro de reconstrução por feature
│   │
│   └── serving/
│       ├── alerts.py         # Dispatch: Console, Telegram Bot, Webhook Defesa Civil
│       ├── pipeline.py       # Orquestrador do pipeline de inferência horária
│       └── api.py            # Servidor FastAPI (endpoints: /predict, /metrics, /drift)
│
├── docker/
│   ├── Dockerfile            # Imagem Python com FastAPI e dependências
│   └── docker-compose.yml    # Orquestração do container de serving
│
├── verify_api.py             # Script de validação dos endpoints da API
├── requirements.txt          # Dependências Python
└── README.md
```

---

## Instalação e Execução

### Pré-requisitos
- Python 3.10+
- Conda (recomendado) ou pip
- ~2 GB de espaço em disco (modelo + dados)

### 1. Criar e ativar o ambiente

```bash
# Com Conda (recomendado)
conda create -n hexclima python=3.10
conda activate hexclima

# Com pip puro
python -m venv .venv && .venv\Scripts\activate  # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Gerar dados

```bash
# Dados meteorológicos (INMET simulado: 2018–2024)
conda run -n hexclima python -m src.data.fetch_inmet

# Dados hidrológicos (ANA/CPRM simulado, baseado na precipitação)
conda run -n hexclima python -m src.data.fetch_ana

# Merge, NaN handling, RobustScaler, partições treino/val/teste
conda run -n hexclima python -m src.data.preprocess
```

### 4. Treinar o modelo

```bash
# Treina nas janelas normais de 2018–2022
# Calibra thresholds sazonais na validação de 2023
conda run -n hexclima python -m src.model.train
```

> O treinamento completo (50 épocas) leva ~60–90 minutos em CPU. Os artefatos serão salvos em `models/`.

### 5. Iniciar a API

```bash
conda run -n hexclima python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

Acesse a documentação interativa: **http://127.0.0.1:8000/docs**

---

## API FastAPI

### `GET /`
Verifica status do servidor e se modelo/scaler/thresholds estão carregados.

### `POST /predict`
Recebe uma janela de **exatamente 72 horas** de dados e retorna o diagnóstico.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "records": [
         {
           "timestamp": "2024-05-01 00:00:00",
           "precipitacao": 15.0,
           "nivel_rio": 450.0,
           "velocidade_vento": 28.5,
           "temperatura": 16.4,
           "umidade": 98.2,
           "pressao": 995.1,
           "vazao": 1200.0
         }
         ... (72 registros no total)
       ]
     }'
```

**Resposta:**
```json
{
  "prediction": {
    "level": "EMERGÊNCIA",
    "reconstruction_error": 1.2451,
    "threshold": 0.6081,
    "severity": 1.0478,
    "is_anomaly": true,
    "season": "outono"
  },
  "explanation": [
    { "feature": "nivel_rio",    "reconstruction_error": 0.412, "contribution_pct": 45.2 },
    { "feature": "precipitacao", "reconstruction_error": 0.231, "contribution_pct": 25.3 },
    { "feature": "vazao",        "reconstruction_error": 0.198, "contribution_pct": 21.7 }
  ]
}
```

### `GET /metrics?dataset=validation|test`
Retorna métricas de classificação (F2-score, AUROC) e operacionais (Lead Time em horas).

### `POST /drift`
Detecta desvio estatístico (KL-divergence) nas features em relação ao conjunto de treino.
Retorna `recommend_retraining: true` se ≥ 2 features apresentarem drift significativo.

---

## Níveis de Alerta

Alinhados ao protocolo da **Defesa Civil do Rio Grande do Sul**:

| Nível | Threshold | Ação recomendada |
|---|---|---|
| 🟢 **NORMAL** | erro < p95 | Monitoramento de rotina |
| 🟡 **ATENÇÃO** | p95 ≤ erro < p97 | Notificar equipe técnica, aumentar frequência de leitura |
| 🟠 **ALERTA** | p97 ≤ erro < p99 | Acionar Defesa Civil, pré-posicionar equipes |
| 🔴 **EMERGÊNCIA** | erro ≥ p99 | Evacuar áreas de risco, ativar protocolo de crise |

---

## Métricas de Avaliação

| Métrica | Prioridade | Justificativa |
|---|---|---|
| **F2-score** | 🥇 Principal | Contexto de vida/morte — recall vale 2× mais que precision |
| **Lead Time (h)** | 🥇 Principal | Mínimo operacional: 6h antes do pico (padrão Defesa Civil RS) |
| **False Alarm Rate** | 🥈 Secundária | Alarmes falsos destroem a credibilidade do sistema |
| **AUROC / AUPRC** | 🥈 Secundária | Independente de threshold, ideal para comparar variantes |
| **Accuracy** | 🥉 Ignorar | Inútil com classes fortemente desbalanceadas |

---

## Docker

```bash
# Build e start do container da API
docker-compose -f docker/docker-compose.yml up --build -d

# A API estará disponível em http://localhost:8000
```

Os diretórios `data/` e `models/` são montados como volumes persistentes.

---

## Roadmap

### v1 — Atual ✅
- [x] LSTM Autoencoder com detecção de anomalia por erro de reconstrução
- [x] Thresholds sazonais (verão/outono/inverno/primavera)
- [x] API FastAPI com `/predict`, `/metrics`, `/drift`
- [x] Sistema de alertas (Console, Telegram, Webhook)
- [x] Explicabilidade por feature (atribuição do erro de reconstrução)

### v2 — Forecasting ⏳
- [ ] **Decoder futuro:** prever as próximas 2h/6h/12h de condições
- [ ] **Score de anomalia futura:** "probabilidade de anomalia nas próximas N horas"
- [ ] Retreinamento automático via loop de feedback pós-evento

### v3 — Espacial / Hiperlocal 🗺️
- [ ] Integração com Modelo Digital de Terreno (MDT/LiDAR)
- [ ] Mapa de risco por segmento de rua (probabilidade + tempo estimado)
- [ ] GNN (Graph Neural Network) sobre rede de drenagem urbana
- [ ] Calibração por bacia hidrográfica individual

---

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. ICML Workshop.
- CPRM (2024). *Boletim de Monitoramento Hidrológico — Enchente RS Maio/2024*.
- CEMADEN (2024). *Sistema de Alertas de Desastres Naturais*.
- ERA5 Reanalysis — Copernicus Climate Change Service (C3S).
