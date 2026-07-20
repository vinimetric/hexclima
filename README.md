#

<img alt="HexClima" src="https://github.com/user-attachments/assets/e17b9de3-fb9b-4809-966c-9b020cb5429d" />

### Monitoramento de Eventos Climáticos Extremos no Rio Grande do Sul

> Detecção de anomalias e previsão de curto prazo via LSTM Autoencoder com dados reais INMET

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Sumário

1. [O que é o HexClima](#o-que-é-o-hexclima)
2. [Capacidades e Limitações](#capacidades-e-limitações)
3. [Arquitetura do Modelo](#arquitetura-do-modelo)
4. [Features e Dados](#features-e-dados)
5. [Particionamento e Paradigma de Treino](#particionamento-e-paradigma-de-treino)
6. [Estrutura do Projeto](#estrutura-do-projeto)
7. [Instalação e Execução](#instalação-e-execução)
8. [API FastAPI — Endpoints](#api-fastapi--endpoints)
9. [Níveis de Alerta](#níveis-de-alerta)
10. [Métricas de Avaliação](#métricas-de-avaliação)
11. [Docker](#docker)
12. [Roadmap](#roadmap)
13. [Referências](#referências)

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## O que é o HexClima

O **HexClima** é um sistema de monitoramento climático em tempo real. Um **LSTM Autoencoder Joint** aprende os padrões normais de comportamento meteorológico e hidrológico a partir dos dados reais das estações INMET (BDMEP). Quando o padrão atual desvia significativamente do que o modelo aprendeu como "normal", o sistema:

1. **Classifica** a situação atual em um dos quatro níveis de alerta através de Análise de Reconstrução
2. **Prevê** as próximas 12 a 24 horas de condições meteorológicas e hidrológicas reais através de um **Forecaster Realista** (treinado inclusive com eventos extremos)
3. **Avalia proativamente** se a janela futura prevista constitui uma anomalia (passando a previsão pelo AE) — gerando um alerta futuro com lead time explícito

> **Paradigma crítico — Treino exclusivamente em dados normais:** Todos os períodos de eventos extremos conhecidos são **explicitamente removidos** do conjunto de treino antes da normalização. O `RobustScaler` é ajustado **somente** no conjunto de treino limpo. Isso garante que o alto erro de reconstrução seja um sinal genuíno de desvio do padrão normal — e não um padrão aprendido.

### Eventos de validação

- 🌊 **Enchente Vale do Taquari — Setembro 2023** (conjunto de validação)
- 🌊 **Grande Enchente RS — Abril/Maio 2024** — o maior desastre climático da história do estado (conjunto de teste)

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Capacidades e Limitações

### Implementado

| Capacidade | Detalhe |
|---|---|
| **Dados reais BDMEP-INMET** | Ingestão dos CSVs das estações automáticas, mapeamento de colunas, timestamps de HHMM |
| **Simulação hidrológica acoplada** | Nível do rio e vazão gerados via filtro autorregressivo físico sobre a precipitação real |
| **Detecção de anomalia (72h)** | Erro MAE de reconstrução comparado a threshold sazonal (p97 por estação do ano) |
| **Forecasting (12h)** | Decoder futuro paralelo prevê as próximas 12h de todas as features |
| **Alerta futuro antecipado** | Janela futura (passado recente + previsão) reavaliada pelo AE — `future_anomaly` com `lead_time_hours` |
| **4 níveis de alerta** | NORMAL / ATENÇÃO / ALERTA / EMERGÊNCIA, alinhados à Defesa Civil RS |
| **Thresholds sazonais** | Percentil 97 calculado separadamente por estação (verão/outono/inverno/primavera) |
| **Explicabilidade por feature** | Contribuição percentual de cada variável no erro de reconstrução |
| **Detecção de data drift** | KL-divergence por feature vs. distribuição de treino; `recommend_retraining` se ≥ 2 features em drift |
| **API REST FastAPI** | Endpoints `/`, `/predict`, `/metrics`, `/drift` com documentação Swagger |
| **Sistema de alertas** | Dispatch para Console, Telegram Bot e Webhook da Defesa Civil |

### Fora do escopo atual

| Limitação | Explicação |
|---|---|
| **Mapa de risco por rua** | Sem componente espacial/GIS — não indica qual área específica vai alagar |
| **Tempo até alagamento** | Não estima "Rua X alaga em ~35 min" (requer dados geoespaciais + modelo supervisionado) |
| **Hidrologia real (ANA/CPRM)** | Nível e vazão são simulados; integração com dados reais da ANA está no roadmap |
| **Inferência hiperlocal** | Sem Modelo Digital de Terreno (MDT/LiDAR) integrado |

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Arquitetura do Modelo (Abordagem Desacoplada)

O sistema utiliza dois modelos trabalhando em sinergia:
1. **Anomaly Detector (LSTM Autoencoder)**: Treinado **exclusivamente em dados normais** para reconhecer padrões saudáveis e detectar desvios (Enchentes).
2. **Realistic Forecaster (Seq2Seq LSTM)**: Treinado em **toda a base de dados** (incluindo as anomalias e enchentes) para aprender a correlação hidrológica extrema (chuva -> inundação) e prever o futuro real.

### Diagrama da Sinergia

```
[ Mundo Real (t-72 até t) ]
             |
      +------+------+
      |             |
[LSTM Autoenc.] [Forecaster]  <-- Previsão realista de (t até t+12)
(Apenas normal) (Treino c/ Enchentes)
      |             |
[Erro Atual]  [ Janela Prevista ]
                    |
              [LSTM Autoenc.] <-- Analisa se o futuro será anômalo!
                    |
              [Erro Futuro]
```

**Detecção de anomalia presente (AE):**
```
MAE = mean(|Input_Real - AE(Input_Real)|)
MAE > threshold_sazonal  -->  ANOMALIA
```

**Alerta futuro (lead time = 12h):**
```
janela_futura = Input_Real[-60:] + Previsao_Forecaster(12h)
MAE_futuro = mean(|janela_futura - AE(janela_futura)|)
MAE_futuro > threshold_sazonal  -->  ANOMALIA FUTURA GERALMENTE ANTES DO EVENTO
```

### Variantes disponíveis

| Variante | Encoder | Uso |
|---|---|---|
| `classic` | LSTM(128) → LSTM(64) | Padrão — melhor custo/benefício |
| `bidirectional` | BiLSTM(64) → BiLSTM(32) | +precisão, mais lento |
| `conv_lstm` | Conv1D(32) + MaxPool + LSTM(64) | Padrões locais diários/semanais |

> O parâmetro `future_steps` em `config.yaml` controla se o decoder de forecasting é ativado. Se `future_steps: 0`, apenas o decoder de reconstrução é usado.

### Hiperparâmetros (`config/config.yaml`)

| Parâmetro | Valor |
|---|---|
| `timesteps` | 72 horas |
| `n_features` | 9 |
| `future_steps` | 12 horas |
| `epochs` | 50 (com EarlyStopping patience=5) |
| `batch_size` | 64 |
| `learning_rate` | 0.001 (com ReduceLROnPlateau) |
| `global_percentile` | 97 |
| `seasonal_percentile` | 97 |
| `drift_threshold_kl` | 0.1 |

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Features e Dados

### Features do modelo (9 variáveis)

| Feature | Origem | Descrição |
|---|---|---|
| `precipitacao` | INMET real | Precipitação total horária (mm) |
| `nivel_rio` | Simulado | Nível do rio (cm) — curva de chaveamento sobre vazão |
| `velocidade_vento` | INMET real | Velocidade do vento (km/h, convertida de m/s) |
| `temperatura` | INMET real | Temperatura do ar — bulbo seco (°C) |
| `umidade` | INMET real | Umidade relativa do ar (%) |
| `pressao` | INMET real | Pressão atmosférica ao nível da estação (mB) |
| `vazao` | Simulado | Vazão fluvial (m³/s) — filtro autorregressivo α=0.985 |
| `precip_24h` | Engenharia | Precipitação acumulada nas últimas 24h (rolling sum) |
| `nivel_rio_ma_48h` | Engenharia | Média móvel do nível do rio nas últimas 48h (tendência) |

### Estações INMET (BDMEP)

| Código | Nome / Localização | Período disponível | Status |
|---|---|---|---|
| **A801** ⭐ | Porto Alegre — Jardim Botânico | 2022-01-01 → 2026-06-26 | Operante — **primária** |
| B807 | Porto Alegre — Belém Novo | 2022-12-07 → 2025-10-19 | Desativada |
| B825 | Porto Alegre — Belém Novo | 2025-04-30 → 2026-06-26 | Operante |

> A estação ativa é controlada por `station_code: "A801"` no `config.yaml`.

### Formato dos arquivos BDMEP

Os arquivos seguem o padrão `dados_<CODIGO>_H_<DATA_INI>_<DATA_FIM>.csv`:
- Separador: `;`
- Encoding: `latin1`
- Primeiras **10 linhas**: metadados (puladas automaticamente)
- Coluna de hora: `Hora Medicao` no formato `HHMM` (inteiro: `0`, `100`, `1200`, `2300`)

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Particionamento e Paradigma de Treino

> ⚠️ **Regra crítica:** O AE aprende **somente o padrão normal**. Incluir eventos anômalos no treino ensinaria o modelo a reconstruí-los bem, zerando a capacidade de detecção.

### Partição temporal (dados reais 2022–2026)

```
Jan/2022 ──────────────── Ago/2023  │  Set/2023 ─ Dez/2023  │  Jan/2024 ──────── Jun/2026
          TREINO                     │  VALIDAÇÃO              │  TESTE
   14.592h — 100% normal             │  2.928h                 │  21.792h
   Expurgo de anomalias aplicado     │  216h anomalas          │  552h anomalas
                                     │  Enchente Set/2023      │  Grande Enchente Mai/2024
```

### Períodos expurgados do treino (`ANOMALY_PERIODS` em `preprocess.py`)

| Evento | Período removido | Horas |
|---|---|---|
| Enchente Vale do Taquari | 2023-09-02 → 2023-09-10 | 216h |
| Grande Enchente RS | 2024-04-28 → 2024-05-20 | 552h |

> Como o treino vai até `2023-08-31`, nenhum desses períodos cai no treino. O expurgo é uma salvaguarda explícita para casos futuros onde a janela de treino for expandida.

### Normalização

O `RobustScaler` é ajustado **somente** no conjunto de treino (`fit_transform`) e depois aplicado à validação e ao teste (`transform`). Isso evita data leakage.

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Estrutura do Projeto

```
lstm_autoencoder_rs/
│
├── config/
│   └── config.yaml              # Todos os hiperparâmetros, features, caminhos e thresholds
│
├── data/
│   ├── BDMEP-INMET_raw/         # CSVs brutos do INMET (formato BDMEP)
│   ├── raw/                     # inmet.csv e ana.csv (pós-ingestão)
│   ├── processed/               # train/val/test .csv normalizados e rotulados
│   └── events/                  # historical_events.csv (referência)
│
├── models/
│   ├── lstm_ae_rs_v1.h5         # Modelo Keras salvo (melhor checkpoint)
│   ├── scaler_rs_v1.pkl         # RobustScaler serializado (fit no treino)
│   └── thresholds_rs_v1.json    # Thresholds p95/p97/p99 globais e sazonais
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingest_real_data.py  # Ingestão BDMEP + simulação hidrológica autorregressiva
│   │   ├── fetch_inmet.py       # Gerador sintético (fallback)
│   │   ├── fetch_ana.py         # Gerador sintético (fallback)
│   │   └── preprocess.py        # Merge, engenharia de features, expurgo, RobustScaler, partições
│   │
│   ├── model/
│   │   ├── architecture.py      # build_lstm_autoencoder() — 3 variantes, suporte a future_steps
│   │   ├── train.py             # Loop multi-tarefa, EarlyStopping, calibração sazonal de thresholds
│   │   └── anomaly.py           # detect_anomaly() — reconstrução + forecasting + alerta futuro
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # F2-score, FAR, AUROC, AUPRC, Lead Time operacional
│   │   └── explainability.py    # Atribuição percentual do erro por feature
│   │
│   └── serving/
│       ├── alerts.py            # dispatch_alert() — Console, Telegram, Webhook Defesa Civil
│       ├── pipeline.py          # inference_pipeline() — orquestrador das 72h → detect_anomaly
│       └── api.py               # FastAPI — /, /predict, /metrics, /drift
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── verify_api.py                # Script de smoke-test dos endpoints
├── requirements.txt
└── README.md
```

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Instalação e Execução

### Pré-requisitos

- Python 3.10+, TensorFlow 2.12+
- Conda com ambiente `watenv` (ou outro com TF instalado)
- ~2 GB de disco (modelo + dados processados)

### 1. Ativar o ambiente

```bash
conda activate watenv
```

### 2. Ingestão dos dados reais BDMEP-INMET

Coloque os arquivos CSV em `data/BDMEP-INMET_raw/` com o nome padrão:
`dados_A801_H_2022-01-01_2026-06-26.csv`

```bash
# Windows PowerShell — define encoding para evitar erros de caracteres especiais do TF
$env:PYTHONIOENCODING='utf-8'

# Lê o CSV bruto, mapeia colunas, cria timestamps, simula hidrologia
conda run -n watenv python -m src.data.ingest_real_data
```

> **Sem dados reais?** Use os geradores sintéticos como fallback:
> ```bash
> conda run -n watenv python -m src.data.fetch_inmet
> conda run -n watenv python -m src.data.fetch_ana
> ```

### 3. Pré-processamento

```bash
# Merge, engenharia de features (precip_24h, nivel_rio_ma_48h),
# expurgo de anomalias, RobustScaler, partições treino/val/teste
$env:PYTHONIOENCODING='utf-8'
conda run -n watenv python -m src.data.preprocess
```

**Saída esperada:**
```
Registros apos expurgo - Treino: 14592, Validacao: 2928, Teste: 21792
  Treino cobre: 2022-01-01 00:00:00 ate 2023-08-31 23:00:00
  Validacao cobre: 2023-09-01 00:00:00 ate 2023-12-31 23:00:00
  Teste cobre: 2024-01-01 00:00:00 ate 2026-06-26 23:00:00
Rotulos: 216 horas anomalas na validacao, 552 no teste
```

### 4. Treinar o modelo

```bash
$env:PYTHONIOENCODING='utf-8'
conda run -n watenv python -m src.model.train
```

O script:
- Constrói o LSTM AE Joint (`classic`, `future_steps=12`)
- Compila com duas losses MAE: `output_recon` e `output_fore` (pesos iguais = 1.0)
- Treina com `EarlyStopping(patience=5)` + `ReduceLROnPlateau(patience=3)` + `ModelCheckpoint`
- Valida **somente** nas janelas normais da validação (sem anomalias marcadas)
- Calibra thresholds sazonais (p97) e salva `thresholds_rs_v1.json`

### 5. Iniciar a API

```bash
$env:PYTHONIOENCODING='utf-8'
conda run -n watenv uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

Documentação interativa: **http://127.0.0.1:8000/docs**

### 6. Validar os endpoints

```bash
conda run -n watenv python verify_api.py
```

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## API FastAPI — Endpoints

### `GET /`

Verifica se modelo, scaler e thresholds estão carregados.

```json
{
  "status": "online",
  "model_loaded": true,
  "scaler_loaded": true,
  "thresholds_loaded": true,
  "api_name": "LSTM Autoencoder RS Anomaly Detection API"
}
```

---

### `POST /predict`

Recebe exatamente **72 registros horários consecutivos** e retorna diagnóstico completo.

O payload deve conter os **7 campos brutos** (as 2 features de engenharia são calculadas internamente pelo pipeline):

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
         // ... (72 registros no total)
       ]
     }'
```

**Resposta:**
```json
{
  "prediction": {
    "timestamp": "2024-05-01 23:00:00",
    "reconstruction_error": 1.2451,
    "threshold": 0.6081,
    "severity": 1.0478,
    "level": "EMERGÊNCIA",
    "is_anomaly": true,
    "season": "outono",
    "forecast": [
      {
        "timestamp": "2024-05-02 00:00:00",
        "precipitacao": 18.4,
        "nivel_rio": 467.2,
        "velocidade_vento": 31.0,
        "temperatura": 15.9,
        "umidade": 99.1,
        "pressao": 992.3,
        "vazao": 1340.5,
        "precip_24h": 210.3,
        "nivel_rio_ma_48h": 458.0
      }
      // ... (12 registros)
    ],
    "future_anomaly": {
      "reconstruction_error": 1.3812,
      "threshold": 0.6081,
      "severity": 1.2712,
      "level": "EMERGÊNCIA",
      "is_anomaly": true,
      "season": "outono",
      "lead_time_hours": 12
    }
  },
  "explanation": [
    { "feature": "nivel_rio",    "reconstruction_error": 0.412, "contribution_pct": 45.2 },
    { "feature": "precipitacao", "reconstruction_error": 0.231, "contribution_pct": 25.3 },
    { "feature": "vazao",        "reconstruction_error": 0.198, "contribution_pct": 21.7 }
  ]
}
```

**Classificação de `severity`:**
- `severity < 0` → NORMAL (erro abaixo do threshold)
- `0 ≤ severity < 0.2` → ATENÇÃO (até 20% acima)
- `0.2 ≤ severity < 0.6` → ALERTA (entre 20% e 60% acima)
- `severity ≥ 0.6` → EMERGÊNCIA (mais de 60% acima)

---

### `GET /metrics?dataset=validation|test`

Calcula métricas sobre os dados rotulados salvos. Retorna:

```json
{
  "dataset": "test",
  "classification_metrics": {
    "precision": 0.87,
    "recall": 0.94,
    "f1": 0.90,
    "f2": 0.92,
    "far": 0.03,
    "auroc": 0.97,
    "auprc": 0.91,
    "confusion_matrix": [[21100, 650], [33, 519]]
  },
  "operational_metrics": {
    "mean_lead_hours": 14.5,
    "median_lead_hours": 13.0,
    "min_lead_hours": 6.2,
    "events_detected": 1,
    "events_missed": 0
  }
}
```

Picos de referência: `2023-09-05 12:00` (validação), `2024-05-04 12:00` (teste).

---

### `POST /drift`

Detecta desvio estatístico (KL-divergence) por feature em relação ao conjunto de treino.

```json
{
  "drift_scores": {
    "precipitacao": { "kl_divergence": 0.032, "drift_detected": false },
    "nivel_rio":    { "kl_divergence": 0.213, "drift_detected": true }
  },
  "total_features_with_drift": 2,
  "recommend_retraining": true
}
```

`recommend_retraining: true` quando ≥ 2 features com `kl_divergence > 0.1`.

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Níveis de Alerta

Alinhados ao protocolo da **Defesa Civil do Rio Grande do Sul**:

| Nível | Condição | Ação recomendada |
|---|---|---|
| 🟢 **NORMAL** | `erro < threshold` (`severity < 0`) | Monitoramento de rotina |
| 🟡 **ATENÇÃO** | `severity < 0.2` | Notificar equipe técnica; aumentar frequência de leitura |
| 🟠 **ALERTA** | `0.2 ≤ severity < 0.6` | Acionar Defesa Civil; pré-posicionar equipes |
| 🔴 **EMERGÊNCIA** | `severity ≥ 0.6` | Protocolo de crise; evacuação preventiva |

> O campo `future_anomaly.level` informa o **nível previsto para as próximas 12 horas**, permitindo acionamento proativo antes do pico.

Os thresholds sazonais são o percentil 97 do erro MAE nas janelas **normais** da validação, calculados separadamente para:
- **Verão** (dez/jan/fev) — período chuvoso, threshold mais alto
- **Outono** (mar/abr/mai) — transição, inclui ciclo de enchentes históricas
- **Inverno** (jun/jul/ago) — seco, threshold mais baixo
- **Primavera** (set/out/nov) — inclui Enchente Set/2023

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Métricas de Avaliação

| Métrica | Prioridade | Justificativa |
|---|---|---|
| **F2-score** | 🥇 Principal | Recall vale 2× (β=2) — custo de falso negativo é alto em contexto de desastres |
| **Lead Time (h)** | 🥇 Principal | Mínimo operacional: ≥ 6h antes do pico (padrão Defesa Civil RS) |
| **False Alarm Rate (FAR)** | 🥈 Secundária | Alarmes falsos frequentes invalidam a credibilidade e o protocolo |
| **AUROC / AUPRC** | 🥈 Secundária | Independentes de threshold; ideais para comparar variantes |
| **Accuracy** | ❌ Ignorar | Inútil com classes fortemente desbalanceadas (~2% anomalias) |

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Docker

```bash
# Build e start
docker-compose -f docker/docker-compose.yml up --build -d

# API disponível em http://localhost:8000
```

Os diretórios `data/` e `models/` são montados como volumes persistentes no container.

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Roadmap

### v1 — Base ✅
- [x] LSTM Autoencoder com detecção de anomalia por erro MAE de reconstrução
- [x] Thresholds sazonais (p97 por estação do ano)
- [x] API FastAPI com `/`, `/predict`, `/metrics`, `/drift`
- [x] Sistema de alertas — Console, Telegram, Webhook Defesa Civil
- [x] Explicabilidade por feature (contribuição percentual no erro de reconstrução)

### v2 — Dados Reais + Forecasting ✅
- [x] Ingestão real BDMEP-INMET (`ingest_real_data.py`) — parsing do formato BDMEP, timestamps HHMM, conversão de unidades
- [x] Simulação hidrológica acoplada à precipitação real (filtro AR, curva de chaveamento)
- [x] Engenharia de features: `precip_24h` (rolling sum 24h) e `nivel_rio_ma_48h` (média móvel 48h)
- [x] Expurgo explícito de anomalias do treino do AE (`ANOMALY_PERIODS` em `preprocess.py`)
- [x] Partição temporal correta para dados reais 2022–2026 (treino < Set/2023)
- [x] LSTM AE Clássico para extração de Reconstruction Error e Thresholding sazonal
- [x] Arquitetura desacoplada: Criação do *Realistic Forecaster* Seq2Seq treinado em base total
- [x] Sinergia AE + Forecaster: Alerta futuro (`future_anomaly`) na API avaliando previsão realista

### v3 — Dados Hidrológicos Reais + Espacial 🗺️
- [ ] Integração com dados reais da ANA/CPRM (nível e vazão observados)
- [ ] Integração com Modelo Digital de Terreno (MDT/LiDAR)
- [ ] Mapa de risco por segmento de rua
- [ ] GNN (Graph Neural Network) sobre rede de drenagem urbana
- [ ] Retreinamento automático via loop de feedback pós-evento

<img alt="divider" src="https://github.com/user-attachments/assets/3b2a214d-ddb3-4507-9ed4-9575e30528a7" />

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. ICML Workshop on Anomaly Detection.
- CPRM (2024). *Boletim de Monitoramento Hidrológico — Enchente RS Maio/2024*.
- CEMADEN (2024). *Sistema de Alertas de Desastres Naturais — bdmep.inmet.gov.br*.
- INMET / BDMEP (2024). *Banco de Dados Meteorológicos para Ensino e Pesquisa*. https://bdmep.inmet.gov.br
- ERA5 Reanalysis — Copernicus Climate Change Service (C3S).
