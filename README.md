#

![Description of image](https://private-user-images.githubusercontent.com/43424669/612519115-e17b9de3-fb9b-4809-966c-9b020cb5429d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE0NTEsIm5iZiI6MTc4MjMxMTE1MSwicGF0aCI6Ii80MzQyNDY2OS82MTI1MTkxMTUtZTE3YjlkZTMtZmI5Yi00ODA5LTk2NmMtOWIwMjBjYjU0MjlkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MjU1MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYyNWQ2ZTA1YTE0OTg5YTViNzVlZTYxOTliMzQ1NzNmMGE5Y2M5YWZmZjAzMDY5NTJlNGNkMzU3OWIwM2U4NDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.esGAX0WSelE9pEPepGtPvYr8C9GW5qVlpETwZRTeP1M)

### Detecção / Monitoramento de Eventos Climáticos Extremos

> **Paradigma:** Aprendizado de normalidade, nesta versão o modelo é treinado **exclusivamente** em períodos climáticos normais e detecta eventos extremos (enchentes, temporais) pelo alto **erro de reconstrução** das séries temporais meteorológicas e hidrológicas.

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Sumário

<style>
  .sumario-container {
    font-family: Arial, sans-serif;
    line-height: 1.6;
  }
  .link-verde { 
    color: #6de18b; /* Verde padrão */
    text-decoration: none; 
    font-weight: 500;
    transition: color 0.2s ease, text-decoration 0.2s ease;
  }
  .link-verde:hover { 
    color: #eeeeee; /* Verde mais escuro ao passar o mouse */
    text-decoration: none; 
  }
  .lista-sumario {
    list-style-type: none;
    padding-left: 0;
  }
  .lista-sumario li {
    margin-bottom: 8px;
  }
  .numero-item {
    color: #757575;
    font-weight: bold;
    margin-right: 5px;
  }
</style>

<div class="sumario-container">
  <ol class="lista-sumario">
    <li><span class="numero-item">1.</span><a href="#o-que-é-o-memclimate" class="link-verde">O que é o HexClimate</a></li>
    <li><span class="numero-item">2.</span><a href="#o-que-o-modelo-faz-e-o-que-não-faz" class="link-verde">O que o modelo faz?</a></li>
    <li><span class="numero-item">3.</span><a href="#arquitetura" class="link-verde">Arquitetura</a></li>
    <li><span class="numero-item">4.</span><a href="#fontes-de-dados" class="link-verde">Datasets</a></li>
    <li><span class="numero-item">5.</span><a href="#estrutura-do-projeto" class="link-verde">Estrutura do Projeto</a></li>
    <li><span class="numero-item">6.</span><a href="#instalação-e-execução" class="link-verde">Executando o Projeto</a></li>
    <li><span class="numero-item">7.</span><a href="#api-fastapi" class="link-verde">API FastAPI</a></li>
    <li><span class="numero-item">8.</span><a href="#níveis-de-alerta" class="link-verde">Níveis de Alerta</a></li>
    <li><span class="numero-item">9.</span><a href="#métricas-de-avaliação" class="link-verde">Métricas de Avaliação</a></li>
    <li><span class="numero-item">10.</span><a href="#docker" class="link-verde">Docker</a></li>
    <li><span class="numero-item">11.</span><a href="#roadmap" class="link-verde">Roadmap</a></li>
    <li><span class="numero-item">12.</span><a href="#referências" class="link-verde">Referências</a></li>
  </ol>
</div>

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## O que é o HexClima

O **HexClima** é um sistema de monitoramento climático em tempo real baseado em aprendizado de máquina para o Rio Grande do Sul. Ele utiliza um **LSTM Autoencoder** que aprende os padrões normais de comportamento climático e hidrológico ao longo de anos de dados históricos. Quando o padrão atual desvia significativamente do que o modelo aprendeu como "normal", um alerta é disparado.


### Evento de validação
O sistema foi validado contra a **Grande Enchente do RS de Maio de 2024** — o maior desastre climático da história do estado, afetando mais de 220 municípios.

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## O que o modelo faz?

### _Capacidades atuais

| Capacidade | Detalhe |
|---|---|
| **Detecção de anomalia regional** | Identifica quando as condições das últimas 72h estão fora do padrão histórico |
| **Alerta antecipado implícito** | Detecta a deterioração gradual das condições **antes** do pico da enchente (lead time emergente) |
| **Classificação de severidade** | Quatro níveis: NORMAL → ATENÇÃO → ALERTA → EMERGÊNCIA |
| **Thresholds sazonais** | Limites adaptados por estação (verão chuvoso vs. inverno seco do RS) |
| **Explicabilidade por feature** | Identifica qual featire (chuva, nível do rio, pressão...) mais contribui para o alerta |
| **Detecção de data drift** | Monitora se a distribuição dos dados mudou e o modelo precisa ser retreinado |
| **API REST em tempo real** | Inferência via FastAPI com documentação interativa (Swagger) |

### _Limitações atuais (fora do escopo v0.1.0)

| Limitação | Explicação |
|---|---|
| **Previsão futura explícita** | O modelo detecta o estado *presente*, não prevê "em 2h haverá anomalia" (requer arquitetura Seq2Seq) |
| **Mapa de risco por rua** | Sem componente espacial/GIS — não indica qual rua específica vai alagar |
| **Tempo até o alagamento** | Não estima "Rua X alaga em ~35 min" (requer dados geoespaciais + modelo supervisionado) |
| **Inferência hiperlocal** | Sem Modelo Digital de Terreno (MDT) integrado |

> Veja o [Roadmap](#roadmap) para as extensões planejadas que cobrem estas capacidades.

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Arquitetura

Em nosso problema, lidamos com dados de séries temporais multivariadas. Dados de séries temporais multivariadas contêm múltiplas variáveis ​​observadas ao longo de um período de tempo. Construimos um autoencoder LSTM sobre essas séries temporais multivariadas para realizar a classificação de eventos raros. Isso é feito utilizando uma abordagem de detecção de anomalias.

Construímos um autoencoder com os dados normais, utilizamos para reconstruir as 72h passadas, se o erro de reconstrução for alto, classificamos a amostra como anômala.


### Diagrama do LSTM Autoencoder
![Description of image](https://private-user-images.githubusercontent.com/43424669/612547167-74e0f848-daa2-4fec-a3ef-34b27a145bfe.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTQ1NjAsIm5iZiI6MTc4MjMxNDI2MCwicGF0aCI6Ii80MzQyNDY2OS82MTI1NDcxNjctNzRlMGY4NDgtZGFhMi00ZmVjLWEzZWYtMzRiMjdhMTQ1YmZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE1MTc0MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI3YWU5NjAwN2IxZWVmOWE1MWFhOTBlMjdiM2VlZTdkNGMzY2FmNWQ4OWZmNjliZTNjZmQ1MGUwMzljNjI0NDAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.7WPv1SHRh6uznHg4lqD1AiDRLEO3K76A2G3LhLDcyX8)
```py

  Erro de Reconstrução (MAE) = |Input - Output|
  
  Se erro > threshold sazonal → ANOMALIA

```

### Uma breve recapitulação sobre LSTM:

A LSTM é um tipo de Rede Neural Recorrente (RNN). As RNNs, de modo geral, e as LSTMs, especificamente, são utilizadas com dados sequenciais ou séries temporais.
Esses modelos são capazes de extrair automaticamente o efeito de eventos passados.
As LSTMs são conhecidas por sua capacidade de extrair efeitos tanto de longo quanto de curto prazo de eventos passados.



### Variantes disponíveis

| Variante | Arquivo | Quando usar |
|---|---|---|
| `classic` | `architecture.py` | Padrão — melhor custo/benefício |
| `bidirectional` | `architecture.py` | +10–15% precisão, mais lento |
| `conv_lstm` | `architecture.py` | Dados com padrões locais diários/semanais |
> Outras arquiteturas serão exploradas, Veja o [Roadmap](#roadmap) para mais detalhes.

### Janelamento temporal

```
Dado bruto horário  →  Janelas deslizantes de 72h  →  Autoencoder
   (stride = 1h)              (N, 72, 7)
```

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Fontes de Dados

| Fonte | Variáveis | Tipo |
|---|---|---|
| **INMET** | Precipitação, Temperatura, Umidade, Pressão, Vento | Meteorológico |
| **ANA / CPRM** | Nível dos rios, Vazão fluvial | Hidrológico |
| **CEMADEN** | Alertas de desastres naturais | Eventos |
| **ERA5 (ECMWF)** | Reanálise climática global (1940–hoje) | Referência |
| **Dados sintéticos** | Gerados pelo `fetch_inmet.py` e `fetch_ana.py` com física realista | Fallback/Demo |

> **Nota:** Os scripts de ingestão incluem um gerador de dados sintéticos fisicamente consistentes como fallback, permitindo rodar o sistema completo sem acesso às APIs externas.

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

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

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

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

# Com pip
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

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

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

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Níveis de Alerta

Alinhados ao protocolo da **Defesa Civil do Rio Grande do Sul**:

| Nível | Threshold | Exemplo de Ação |
|---|---|---|
| 🟢 **NORMAL** | erro < p95 | Nada a fazer / monitoramento de rotina |
| 🟡 **ATENÇÃO** | p95 ≤ erro < p97 | Notificar equipe técnica, aumentar frequência de leitura |
| 🟠 **ALERTA** | p97 ≤ erro < p99 | Acionar Defesa Civil |
| 🔴 **EMERGÊNCIA** | erro ≥ p99 | Protocolo de crise |

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Métricas de Avaliação

| Métrica | Prioridade | Justificativa |
|---|---|---|
| **F2-score** | 🥇 Principal | Garantir nenhum Falso Negativo|
| **Lead Time (h)** | 🥇 Principal | Mínimo operacional: predição ~6h antes do pico (padrão Defesa Civil RS) |
| **False Alarm Rate** | 🥈 Secundária | Alarmes falsos invalida a credibilidade do sistema |
| **AUROC / AUPRC** | 🥈 Secundária | Independente de threshold, ideal para comparar variantes |
| **Accuracy** | 🥉 Ignorar | Exemplo de metrica que não é util (classes fortemente desbalanceadas) |

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Docker

```bash
# Build e start do container da API
docker-compose -f docker/docker-compose.yml up --build -d

# A API estará disponível em http://localhost:8000
```

Os diretórios `data/` e `models/` são montados como volumes persistentes.

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)
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

![Description of image](https://private-user-images.githubusercontent.com/43424669/612522697-3b2a214d-ddb3-4507-9ed4-9575e30528a7.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODIzMTE3ODgsIm5iZiI6MTc4MjMxMTQ4OCwicGF0aCI6Ii80MzQyNDY2OS82MTI1MjI2OTctM2IyYTIxNGQtZGRiMy00NTA3LTllZDQtOTU3NWUzMDUyOGE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI0VDE0MzEyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBiOGQ3NmIzYmI5YjVjN2Y4MmU4MjhkMzg3MDZkMTY4MzFiMTA1NDQ5Y2Q4ZmI0MDljZjJlMTcyZDc4YjkwODEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.h7394RV9uefR2PwrQmAYb9ZlrpqMVjBgH9PtUQKFUTA)

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. ICML Workshop.
- CPRM (2024). *Boletim de Monitoramento Hidrológico — Enchente RS Maio/2024*.
- CEMADEN (2024). *Sistema de Alertas de Desastres Naturais*.
- ERA5 Reanalysis — Copernicus Climate Change Service (C3S).
