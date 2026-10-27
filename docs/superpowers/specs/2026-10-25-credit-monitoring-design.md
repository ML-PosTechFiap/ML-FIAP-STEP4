# Design — Monitoramento Contínuo de Credit Scoring (Tech Challenge Fase 4)

- **Data:** 2026-10-25
- **Autor:** João Belletati (projeto individual)
- **Status:** aprovado em brainstorming, aguardando revisão da spec
- **Fonte dos requisitos:** `MLET - Tech Challenge - Fase 4.pdf`

## 1. Objetivo

Construir a camada de **sustentação e confiabilidade** de um modelo de Credit Scoring de uma
fintech: contratos de dados que bloqueiam lotes ruins, simulação de degradação ao longo do tempo
(data drift e concept drift), detecção estatística com Evidently, observabilidade de pipeline,
infraestrutura e modelo, e governança documentada sob a LGPD, incluindo análise causal.

### Critérios de avaliação e onde cada um é atendido

| Critério | Peso | Seções |
|---|---|---|
| Validação de dados (Pandera interceptando anomalias) | 25% | §5 |
| Detecção de drift (simulação + relatório Evidently) | 25% | §6 |
| Observabilidade e logs (dashboard + alertas) | 20% | §7 |
| Governança (privacidade, base legal, mitigação de vieses) | 15% | §8 |
| Vídeo STAR (≤ 5 min) | 15% | §9 |

### Fora do escopo

- Retreino automático (fica documentado como prescrição).
- API online de predição (o pipeline é batch mensal).
- Alertmanager e notificações externas (alertas disparam no Prometheus e aparecem no Grafana).
- Tuning extenso de hiperparâmetros.
- Tabelas auxiliares do Home Credit (bureau, previous_application etc.).

## 2. Decisões principais

| Tema | Decisão |
|---|---|
| Dataset | Home Credit Default Risk — **apenas `application_train.csv`** (307.511 linhas) |
| Modelo | `XGBClassifier` dentro de um `Pipeline` do scikit-learn |
| Contrato de dados | **Pandera** (`DataFrameModel`), `lazy=True` |
| Drift | **Evidently** (versão fixada) + PSI/KS próprios (`scipy`) |
| Simulação | 6 lotes mensais com drift progressivo + mês 3 chegando sujo |
| Observabilidade | **MLflow + Prometheus (Pushgateway) + Grafana + Loki** |
| Orquestração | CLI Python (`python -m credit_monitor <comando>`), sem orquestrador |
| Execução | **Tudo containerizado** via `docker compose`; Makefile só como atalho opcional |
| Causalidade | Atribuição empírica: decomposição data vs concept drift + ablação por feature + SHAP |

### Restrições do ambiente

- Host Windows 11 sem `make`: todo comando documentado precisa funcionar como
  `docker compose ...` no PowerShell. O `Makefile` existe só como atalho para Linux/macOS/WSL.
- O download do Kaggle exige conta com as regras da competição aceitas. Dois caminhos são
  suportados: `KAGGLE_USERNAME`/`KAGGLE_KEY` no `.env`, ou o arquivo colocado manualmente em
  `data/raw/application_train.csv`.

### Versões fixadas que diferem do plano

- `pyarrow==17.0.0` (e não 18): `mlflow==2.17.2` exige `pyarrow<18`.
- `multimethod==1.10.0`: `pandera==0.21.0` quebra com `multimethod` 2.x e avisa a partir da 1.11.
- `sqlalchemy==2.0.36` na imagem do MLflow (`infra/mlflow/Dockerfile`): o MLflow 2.17 não importa com
  SQLAlchemy 2.1 ou superior.

## 3. Arquitetura

### 3.1 Serviços (`docker-compose.yml`)

| Serviço | Função | Porta (bind em 127.0.0.1) |
|---|---|---|
| `pipeline` | Imagem Python com o pacote `credit_monitor`. Executado sob demanda: `docker compose run --rm pipeline <comando>` | — |
| `mlflow` | Tracking server (backend SQLite + artefatos em volume), iniciado com `--expose-prometheus` | 5000 |
| `pushgateway` | Recebe as métricas de cada execução batch | 9091 |
| `prometheus` | Coleta Pushgateway e `/metrics` dos demais serviços e avalia `alerts.yml` | 9090 |
| `grafana` | Datasources (Prometheus, Loki) e dashboard provisionados a partir do repositório | 3000 |
| `loki` | Recebe os logs do pipeline via API HTTP (sem agente) | 3100 |

As portas do host podem ser sobrescritas no `.env` (`MLFLOW_PORT`, `PUSHGATEWAY_PORT`, `PROMETHEUS_PORT`,
`LOKI_PORT`, `GRAFANA_PORT`), por exemplo quando outra stack local já ocupa 3000 ou 9090. Os padrões da
tabela acima não mudam, e a comunicação entre contêineres continua nas portas internas.

O container `pipeline` monta `src/`, `data/` e `reports/` como bind mounts, para que editar código
não exija rebuild. Só mudanças de dependência (`pyproject.toml`) pedem `docker compose build`.

### 3.2 Fluxo

```
download ─► prepare ─► train
              │           │  (contrato na referência, treino, registro no MLflow)
              ▼           ▼
      lotes mensais    modelo "production"
              │
              ▼
monitor --batch N:  ingestão ─► contrato ─┬─► (falha) quarentena + métricas + alerta
                     (pseudonimiza)       │
                                          └─► transformação ─► predição ─► drift/desempenho/viés
                                                                               │
                                       publicação: Pushgateway + Loki + MLflow + reports/
```

### 3.3 Comandos do CLI

| Comando | O que faz |
|---|---|
| `download` | Baixa `application_train.csv` via API do Kaggle (ou confere se o arquivo manual existe) |
| `prepare [--synthetic] [--n]` | Aplica o contrato na base bruta (mostra as anomalias reais), limpa a referência, faz o split 60/40 e gera os lotes `01`..`06` e `03-dirty`, cada um com `manifest.json`. Com `--synthetic`, gera antes uma base sintética no mesmo schema (demo e CI sem Kaggle) |
| `train` | Valida a referência, treina, avalia, faz a auditoria de viés e registra `credit-scoring` com alias `production` |
| `monitor --batch <id>` | Roda a esteira mensal para um lote |
| `simulate-all [--interval 30s]` | Roda `01, 02, 03-dirty, 03, 04, 05, 06` em sequência, opcionalmente espaçados |
| `causal` | Gera a análise causal sobre o lote `06` |
| `reset-metrics` | Apaga todos os grupos do Pushgateway (demo limpa) |
| `purge` | Aplica a política de retenção (quarentena e logs) |

### 3.4 Estrutura do repositório

```
Tech_Challenge_4/
├── docker-compose.yml  Dockerfile  pyproject.toml  Makefile  .env.example  .gitignore
├── src/credit_monitor/
│   ├── __main__.py  cli.py  config.py
│   ├── data/            download.py  prepare.py  simulate.py
│   ├── contracts/       schema.py  validate.py
│   ├── model/           features.py  train.py  predict.py  fairness.py
│   ├── drift/           stats.py  detect.py  causal.py
│   ├── privacy/         pseudonymize.py  retention.py
│   ├── observability/   logging.py  metrics.py  tracking.py  stage.py
│   └── pipeline.py      # orquestra monitor / simulate-all
├── infra/
│   ├── prometheus/      prometheus.yml  alerts.yml
│   ├── grafana/         provisioning/  dashboards/credit-monitoring.json
│   └── loki/            loki-config.yml
├── tests/
├── data/                # gitignored: raw/ reference/ batches/ quarantine/
├── reports/             # gitignored: logs/ contracts/ drift/ causal/ index.html
└── docs/
    ├── superpowers/specs/  roteiro-video.md
    └── evidence/        # cópia versionada dos relatórios finais e prints do dashboard
```

### 3.5 Degradação graciosa

Se MLflow, Pushgateway ou Loki não responderem, o pipeline registra um `WARN` e segue. Logs JSONL
e relatórios HTML em `reports/` são sempre gravados. Isso permite rodar `pytest` no CI sem subir a
infraestrutura.

### 3.6 CI (GitHub Actions)

`ruff` + `pytest` (dados sintéticos pequenos, sem o dataset real) + `promtool check rules
infra/prometheus/alerts.yml` usando a imagem oficial do Prometheus.

## 4. Dados e split

### 4.1 Colunas usadas (minimização: `usecols`)

| Grupo | Colunas |
|---|---|
| Identificador | `SK_ID_CURR`: usado só no contrato (unicidade); pseudonimizado na ingestão; nunca vira feature |
| Financeiras | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Perfil | `DAYS_BIRTH`, `DAYS_EMPLOYED`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS` |
| Categóricas | `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY` |
| Score externo | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` (nulos permitidos) |
| Sensível | `CODE_GENDER`: só auditoria de viés, **fora das features** |
| Alvo | `TARGET` |

Features derivadas (`model/features.py`): `AGE_YEARS`, `YEARS_EMPLOYED` (NaN quando sentinela),
`IS_UNEMPLOYED_OR_PENSIONER` (`DAYS_EMPLOYED == 365243`), `CREDIT_INCOME_RATIO`,
`ANNUITY_INCOME_RATIO`, `CREDIT_TERM` (`AMT_CREDIT / AMT_ANNUITY`).

### 4.2 Split (estratificado por `TARGET`, `SEED = 42`)

- **Referência (60%)**: dividida em treino 70% / validação 15% / teste 15%.
- **Pool de produção (40%)**: fonte dos lotes mensais. Cada lote `01`..`06` é um sorteio **independente** do
  pool inteiro (sem reposição dentro do lote; uma linha pode aparecer em meses diferentes), com
  `min(BATCH_SIZE, len(pool) // 6)` linhas (20 mil com os dados reais) e com reamostragem ponderada onde a
  simulação exige.
- A base de comparação do drift é o **teste da referência com predições**, não o treino.

## 5. Etapa 1 — Contrato de dados e modelo base

### 5.1 Contrato (`contracts/schema.py`, `CONTRACT_VERSION = "1.0"`)

Regras rígidas e bloqueantes, aplicadas ao lote **bruto** antes de qualquer transformação:

1. `SK_ID_CURR` não nulo e único.
2. `AMT_INCOME_TOTAL` não nulo, `> 0` e `≤ 10_000_000`.
3. Idade derivada de `DAYS_BIRTH` entre 18 e 100 anos.
4. `DAYS_EMPLOYED ≤ 0` ou `== 365243` (sentinela conhecido).
5. `AMT_CREDIT > 0`; `AMT_ANNUITY` não nulo e `> 0`; `EXT_SOURCE_*` em `[0, 1]` quando presentes.
6. Categóricas no domínio conhecido (domínios definidos a partir da referência limpa; `CODE_GENDER ∈ {M, F}`).
7. Colunas obrigatórias presentes, tipos corretos e lote com pelo menos 1.000 linhas (check de DataFrame).
   O schema aceita a coluna `TARGET` como opcional (`0/1`), para lotes com e sem rótulo.

O contrato valida o frame já pseudonimizado: a regra 1 incide sobre `client_key` (`client_key_unique` e
`client_key_not_null`). O nome de cada check é o rótulo estável `rule`; os erros nativos do Pandera são
normalizados para `column_missing`, `dtype_coercion` e `validation_error` (erro inesperado, que também
bloqueia o lote).

### 5.2 Validação e bloqueio (`contracts/validate.py`)

- `schema.validate(df, lazy=True)` e captura de `SchemaErrors`.
- Em caso de falha, **o lote inteiro é bloqueado**:
  - arquivo **copiado** (não movido) para `data/quarantine/<batch>.parquet`, para que reexecutar o lote bloqueie de novo;
  - `reports/contracts/<batch>_failures.json` e `<batch>_failures.html` com regra, coluna, valor
    ofensor e `client_key`, sem identificador bruto;
  - métricas `contract_passed=0` e `contract_violations{rule}`; run do MLflow com `status=blocked`;
    log `ERROR`;
  - predição e drift **não** são executados para esse lote; o processo retorna código diferente de zero,
    exceto dentro de `simulate-all`, que registra e segue para o próximo lote.

### 5.3 Limpeza da referência

O `prepare` aplica o contrato à base bruta e grava as falhas reais encontradas (por exemplo, renda de
117 milhões, `CODE_GENDER = XNA`, anuidades nulas) em `reports/contracts/raw_reference_failures.json`.
Em seguida remove as linhas ofensoras. A referência limpa precisa passar no contrato sem erros.

### 5.4 Lote sujo `03-dirty`

Cópia do lote `03` com defeitos injetados: renda nula, idade de 15 anos, IDs duplicados, crédito
negativo, categoria desconhecida e `EXT_SOURCE` > 1. O `manifest.json` lista os defeitos e as regras
esperadas.

### 5.5 Modelo base (`model/train.py`)

- `ColumnTransformer`: one-hot nas categóricas (`handle_unknown="ignore"`), numéricas passam direto.
- `XGBClassifier` com `scale_pos_weight = neg/pos`, early stopping na validação e hiperparâmetros
  fixos em `config.py`.
- Métricas no teste: ROC-AUC, PR-AUC, KS e Brier. A expectativa é um ROC-AUC entre 0,74 e 0,76.
- **Ponto de corte** de aprovação escolhido na validação (aprova se a PD for menor que o corte), gravado nos
  metadados do modelo.
- Registro no MLflow: modelo `credit-scoring`, alias `production`, com feature list, corte,
  `CONTRACT_VERSION` e métricas de referência. Cópia local `models/model.joblib` + `metadata.json`
  como fallback.

## 6. Etapa 2 — Simulação e detecção de drift

### 6.1 Cronograma (`config.DRIFT_SCHEDULE`)

| Lote | Inflação: renda / crédito e preço do bem / anuidade | Desempregados (meta) | Novo perfil (jovem, histórico curto) | Concept drift (fração do segmento virada para 1) |
|---|---|---|---|---|
| 01 | 0 / 0 / 0 | referência (cerca de 18%) | não | 0 |
| 02 | +2% / +4% / +6% | referência | não | 0 |
| 03 | +5% / +10% / +14% | 22% | não | 0 |
| 04 | +8% / +16% / +22% | 25% | leve | 10% |
| 05 | +10% / +20% / +28% | 28% | moderado | 18% |
| 06 | +12% / +25% / +35% | 30% | forte | 25% |

- **Inflação:** multiplicação dos valores nominais. As razões derivadas deslocam como consequência.
- **Desemprego:** uma fração de empregados vira `DAYS_EMPLOYED = 365243` e
  `NAME_INCOME_TYPE = "Unemployed"` (categoria já presente na referência, para manter coerência
  entre as colunas e respeitar o domínio do contrato) até atingir a meta. Os empregados restantes têm
  `DAYS_EMPLOYED` encolhido (mais rotatividade).
- **Novo perfil:** reamostragem ponderada do pool favorecendo `AGE_YEARS < 30`, aumento da taxa de
  nulos em `EXT_SOURCE_1` e deslocamento para baixo de `EXT_SOURCE_2/3` (com clip em `[0, 1]`).
- **Concept drift:** no segmento "espremido" (`ANNUITY_INCOME_RATIO` acima do p75 da referência ou
  desempregado), a fração indicada de `TARGET = 0` vira `1`. Os rótulos pré-virada ficam em
  `data/batches/<id>/labels_original.parquet`, invisível para o monitoramento e usado só pelo `causal`.
- Os valores simulados respeitam o contrato: lotes `01`..`06` precisam passar.
- Cada lote grava `manifest.json` com as chaves `batch_id`, `params`, `injected_features`,
  `drifted_features`, `n_rows` e `n_concept_flips` (o `03-dirty` acrescenta `expected_rules`).
  `injected_features` é a verdade-base: toda feature que os parâmetros ativos tocam, detectável ou não.
  `drifted_features` é o subconjunto em que se espera drift detectável (parâmetros acima de um mínimo),
  usado no teste de recall da detecção.
- **Amostragem independente:** cada lote `01`..`06` é sorteado do pool inteiro, e não de um pool que os
  meses anteriores esvaziam. A amostragem disjunta esgotava as linhas jovens e o lote `04` saía com menos
  de 2% de menores de 30 anos (referência: 15%), o que invertia o drift de idade previsto acima.
- Todos os rótulos alterados são sintéticos, e o README declara isso.

**Premissa de rótulos:** o rótulo chega junto com o lote. Em produção real ele chegaria com atraso; essa
premissa fica documentada, e o dashboard separa sinais sem rótulo (drift de dados e de predição) de
sinais com rótulo (desempenho e calibração).

### 6.2 Detecção (`drift/detect.py` e `drift/stats.py`)

- **Evidently:** `DataDriftPreset` com `stattest="psi"` por coluna (features + coluna de score) e
  `ClassificationPreset` (referência vs lote). Saída: `reports/drift/<batch>_data_drift.html`,
  `<batch>_performance.html` e os respectivos JSON.
- **Estatística própria:** PSI com 10 bins por quantil da referência (categóricas por categoria, com
  epsilon para bins vazios) e KS (`ks_2samp`, estatística D) nas numéricas. É a fonte dos alertas; o
  Evidently fornece os relatórios de evidência.
- **Bins do PSI robustos a empates:** cada borda numérica fica no **ponto médio entre o corte do quantil e
  o próximo valor da grade de 100 percentis da referência** acima dele. Com as bordas exatamente sobre os
  valores de empate (renda e preço do bem concentram-se em valores redondos), até uma inflação de 2%
  trocava de bin massas inteiras (PSI da renda 0,49 no lote `02`), um artefato de medição. A correção
  remove esse artefato apenas para deslocamentos menores que a distância entre níveis de preço
  consecutivos; acima disso a massa cruza a borda de verdade, e o PSI do `AMT_GOODS_PRICE` continua não
  monotônico entre lotes.
- **Desempenho por lote:** ROC-AUC, KS, taxa de default observada, PD média prevista, taxa de aprovação e a
  **calibração relativa**: `calibration_ratio_raw = default observado / PD média` e
  `calibration_ratio = calibration_ratio_raw do lote / calibration_ratio_raw da referência` (1,0 = igual à
  referência). É relativa porque o modelo usa `scale_pos_weight`: o score não é uma probabilidade calibrada,
  e o quociente bruto já é cerca de 0,20 na referência.
- **Regra do KS:** usa o tamanho do efeito (D > 0,10), não o p-valor, que satura com cerca de 20 mil linhas.
  O KS é hipersensível em variável discretizada (a renda no lote `02` tem D = 0,119 com PSI 0,001): é drift
  injetado de verdade, e o limiar fica fixo.

### 6.3 Limiares e recomendação

| Sinal | Atenção | Drift / alerta |
|---|---|---|
| PSI por feature | 0,10–0,20 | ≥ 0,20 |
| KS (D) | — | > 0,10 |
| Drift share do dataset | — | ≥ 0,30 |
| PSI do score | — | ≥ 0,20 |
| AUC | — | < AUC de referência − 0,05 |
| Calibração | — | `calibration_ratio` (relativa à referência) > 1,3 |
| Impacto disparato | — | < 0,8 **e** < impacto da referência − 0,05 (`FAIRNESS_DEGRADATION_MARGIN`) |

Recomendação por lote:
- `RETREINAR`: algum alerta de desempenho ou de calibração.
- `INVESTIGAR`: drift de dataset, drift de predição ou alerta de viés, sem queda de desempenho.
- `OK`: nenhum dos casos acima.

**Regra de viés: degradação, não nível.** O impacto disparato por faixa etária já é cerca de 0,54 na
referência (achado do modelo-base, documentado na seção 8.3). Com a regra absoluta (`< 0,8`), `INVESTIGAR`
e o alerta ficariam ligados desde o lote `01`. O alerta e a recomendação disparam só se o impacto do lote
estiver abaixo de 0,8 **e** mais de 0,05 abaixo do impacto do mesmo atributo na referência. O nível-base
exige decisão de governança, não monitoramento mensal.

### 6.4 Evidência

`reports/index.html` liga todos os relatórios e mostra a tabela "features com drift por lote". No fim, uma
cópia dos relatórios e prints vai para `docs/evidence/`.

## 7. Etapa 3 — Observabilidade

### 7.1 Instrumentação (`observability/stage.py`)

Context manager `stage(name, batch)` em volta de cada etapa (`ingest`, `contract`, `transform`,
`predict`, `drift`, `publish`). Ele mede duração, status e linhas de entrada e saída, emite log e registra
métricas.

### 7.2 Logs (`observability/logging.py`)

JSON por linha com `ts, level, run_id, batch_id, stage, event, status, duration_ms, rows_in,
rows_out, message`. São escritos em três destinos: stdout, `reports/logs/pipeline.jsonl` e Loki (handler HTTP em
`/loki/api/v1/push`, labels `job`, `stage`, `level`, `batch`). Nunca contêm dados de linha.

### 7.3 Métricas (`observability/metrics.py`, prefixo `credit_`)

| Grupo | Métricas |
|---|---|
| Pipeline | `pipeline_stage_status{stage,batch}` (1 sucesso, 0 erro, -1 bloqueado pelo contrato), `pipeline_stage_duration_seconds{stage,batch}`, `pipeline_rows_processed{stage,batch}`, `pipeline_last_success_timestamp`, `pipeline_peak_memory_bytes{batch}`, `batch_info{batch_id}` (lote real do grupo `latest`, valor 1) |
| Qualidade | `contract_passed{batch}`, `contract_violations{batch,rule}` |
| Drift | `feature_psi{batch,feature}`, `feature_ks{batch,feature}`, `dataset_drift_share{batch}`, `prediction_psi{batch}` |
| Modelo | `model_auc{batch}`, `model_ks{batch}`, `default_rate_observed{batch}`, `mean_predicted_pd{batch}`, `calibration_ratio{batch}`, `approval_rate{batch}`, `monitor_recommendation{batch}` (0/1/2), `model_auc_reference`, `model_ks_reference` |
| Viés | `fairness_disparate_impact{batch,attribute}` (`gender`, `age_band`) e `fairness_disparate_impact_reference{batch,attribute}` (impacto disparato da referência, para o alerta de degradação e para as barras cinzas do painel) |
| Infra | `up{job}` (coletado pelo Prometheus) e métricas nativas do MLflow, Grafana e Loki |

**Agrupamento no Pushgateway:** cada execução publica no grupo `{job="credit_pipeline", batch=<id>}`
(histórico) **e** no grupo `{job="credit_pipeline", batch="latest"}` (sobrescrito). Os alertas olham só
para `batch="latest"`. O label `batch` vem só da *grouping key* do Pushgateway: as métricas não o declaram.

### 7.4 Alertas (`infra/prometheus/alerts.yml`, `for: 0m`, exceto `InfraTargetDown`)

| Alerta | Expressão (sobre `batch="latest"`) | Severidade |
|---|---|---|
| `DataContractViolation` | `credit_contract_passed == 0` | critical |
| `PipelineStageFailed` | `credit_pipeline_stage_status == 0` (o bloqueio pelo contrato usa -1 e não duplica o `DataContractViolation`) | critical |
| `FeatureDriftDetected` | `credit_feature_psi >= 0.2` | warning |
| `DatasetDriftDetected` | `credit_dataset_drift_share >= 0.3` | critical |
| `PredictionDrift` | `credit_prediction_psi >= 0.2` | warning |
| `ModelPerformanceDegraded` | `credit_model_auc < credit_model_auc_reference - 0.05` | critical |
| `ModelMiscalibrated` | `credit_calibration_ratio > 1.3` | warning |
| `FairnessDegraded` | `credit_fairness_disparate_impact < 0.8 and on(batch, attribute) credit_fairness_disparate_impact < on(batch, attribute) credit_fairness_disparate_impact_reference - 0.05` (queda em relação à referência, não o nível absoluto) | warning |
| `InfraTargetDown` | `up == 0` por 1 minuto | critical |

### 7.5 Dashboard Grafana "Credit Model Monitoring"

1. **Visão executiva:** último lote, recomendação (com mapeamento de cor), alertas disparando, AUC atual vs
   referência e taxa de aprovação.
2. **Qualidade de dados:** status do contrato por lote e violações por regra.
3. **Drift:** heatmap PSI feature × lote, drift share por lote e PSI do score.
4. **Desempenho:** AUC e KS por lote com linha de referência, default observado vs PD prevista e taxa de aprovação.
5. **Viés:** impacto disparato por atributo e lote, com linha em 0,8.
6. **Saúde do pipeline:** duração por etapa, linhas processadas e status das etapas.
7. **Infraestrutura:** alvos no ar, latência e requisições do MLflow, memória de pico do pipeline.
8. **Logs:** painel Loki filtrável por `stage` e `level`.

### 7.6 MLflow

- Experimento `credit-scoring-training`: parâmetros, métricas, auditoria de viés e modelo registrado.
- Experimento `credit-monitoring`: um run por lote.
  - Tags: `batch_id`, `model_version`, `contract_version`, `status`, `recommendation`.
  - Métricas: as mesmas do Pushgateway.
  - Artefatos: HTML/JSON do Evidently, falhas do contrato, `manifest.json` e trecho de log.

### 7.7 Catálogo de métricas

O README traz uma tabela com métrica, tipo, labels, significado, limiar e alerta.

## 8. Etapa 4 — Governança, LGPD, viés e causalidade

### 8.1 Privacy by design / by default (implementado)

| Medida | Implementação |
|---|---|
| Minimização | `usecols` com 21 das 122 colunas (`config.RAW_COLUMNS`) |
| Pseudonimização | `client_key = HMAC-SHA256(PSEUDONYMIZATION_SALT, SK_ID_CURR)` na ingestão; salt no `.env` |
| Observabilidade sem dado individual | Só agregados em logs, métricas, MLflow e Loki; teste automatizado varre as saídas em busca de IDs brutos |
| Atributo de risco fora do modelo | `CODE_GENDER` apenas na auditoria |
| Retenção executável | `purge`: quarentena > 30 dias, logs > 90 dias; Prometheus com `--storage.tsdb.retention.time=395d` |
| Segurança | `data/` e `.env` no `.gitignore`; segredos via `.env`; portas em `127.0.0.1` |

### 8.2 Plano de proteção de dados (README)

- Inventário e classificação das colunas. Gênero não é dado sensível pelo art. 5º, II, mas é vetor de
  discriminação (art. 6º, IX). Sigilo bancário pela LC 105/2001.
- Base legal: art. 7º, X (proteção do crédito) + art. 7º, V. Justificativa para não usar consentimento.
- Decisão automatizada (art. 20): processo de revisão humana e explicação via contribuições SHAP
  (`pred_contribs=True`).
- Direitos do titular (art. 18) e seus limites.
- Retenção:

| Dado | Prazo |
|---|---|
| Dados da proposta | 5 anos após o fim da relação |
| Quarentena | 30 dias |
| Logs | 90 dias |
| Métricas | 13 meses |
| Modelos e relatórios agregados | vida do modelo + 5 anos |

- RIPD resumido (art. 38): riscos × mitigações, encarregado e fluxo de incidentes.
- Ressalva: as citações legais são referência acadêmica, não parecer jurídico.

### 8.3 Viés (`model/fairness.py`)

- Treino: taxa de aprovação e TPR por gênero e por faixa etária (`<30`, `30–45`, `45–60`, `60+`),
  impacto disparato (grupo de menor taxa / grupo de maior taxa) e diferença de equal opportunity.
- Teste de proxy: informação mútua de cada feature com `CODE_GENDER`.
- Monitoramento mensal com `fairness_disparate_impact` e alerta `FairnessDegraded`, que dispara na
  **degradação** em relação à referência (seção 6.3).
- Achado permanente do modelo-base: impacto disparato por faixa etária cerca de 0,54 na referência, abaixo
  do piso de 0,8. Não é drift, e exige decisão de governança antes de ir a produção.
- Mitigações documentadas: ajuste de corte por grupo (com discussão jurídica) ou reponderação no retreino.

### 8.4 Análise causal (`drift/causal.py`, sobre o lote `06`)

1. **DAG causal** em Mermaid no README:
   - inflação → valores nominais → razões;
   - juros → anuidade;
   - desaceleração → desemprego → capacidade real de pagamento (latente) → default;
   - canal digital → perfil jovem com histórico curto → `EXT_SOURCE`.
2. **Decomposição:**
   - AUC_ref;
   - AUC(X₀₆, rótulos originais) = efeito da mudança em X;
   - AUC(X₀₆, rótulos novos) = efeito total;
   - a diferença entre os dois últimos é a contribuição do concept drift.
3. **Ablação por feature:** cada feature com drift volta à distribuição de referência por mapeamento
   de quantis. Medimos a recuperação de PSI do score, AUC e `calibration_ratio`.
4. **Importância:** |SHAP| médio por feature na referência vs `06`.
5. **Saída:** `reports/causal/causal_report.html` + JSON, registrados no MLflow, e conclusão com a prescrição no README:
   retreino com dados recentes, features em termos reais (renda deflacionada), champion/challenger e
   monitoramento das razões de comprometimento.
6. **Ressalva:** as intervenções são válidas porque o processo gerador é controlado pela simulação.

## 9. Vídeo e README

- `docs/roteiro-video.md`: roteiro STAR cronometrado (≤ 5 min) com comandos e painéis por bloco. A gravação
  é feita pelo autor, e o link entra no README.
- Estrutura do README:
  1. Problema.
  2. Arquitetura.
  3. Como rodar (comandos `docker compose` para PowerShell).
  4. Evidências das Etapas 1 a 3.
  5. Catálogo de métricas.
  6. Governança e LGPD.
  7. Viés.
  8. Análise causal.
  9. Lições aprendidas.
  10. Vídeo.

## 10. Testes

| Área | Testes |
|---|---|
| Contrato | DataFrame válido passa; um caso falhando por regra; `03-dirty` falha exatamente nas regras do manifest; referência limpa passa |
| Simulação | Lotes `01`..`06` passam no contrato; parâmetros do manifest refletidos nos dados; `labels_original` preservado |
| Estatística | PSI(x, x) ≈ 0; deslocamento conhecido gera PSI > 0,2; PSI próprio ≈ PSI do Evidently |
| Detecção | Lote `01` sem drift; nos lotes `05`/`06`, recall = 1 sobre as features injetadas |
| Modelo | Smoke test de `train` em amostra sintética; concept drift sintético reduz AUC |
| Observabilidade | `stage()` emite log e métricas com status correto em sucesso e falha; pipeline segue com infraestrutura fora do ar |
| Privacidade | Pseudonimização determinística por salt; nenhum ID bruto em logs, relatórios e artefatos |
| Alertas | `promtool check rules` no CI |

Os testes usam um gerador de dados sintéticos com o mesmo schema do Home Credit, para não depender do
dataset real.

## 11. Commits

Histórico semântico (Conventional Commits), um commit por unidade coerente: `docs:`, `build:`,
`feat(contracts):`, `feat(model):`, `feat(drift):`, `feat(observability):`, `feat(privacy):`,
`test:`, `ci:`.

## 12. Riscos

| Risco | Mitigação |
|---|---|
| API do Evidently muda entre versões | Versão fixada; PSI/KS próprios como segunda camada |
| Drift simulado viola o contrato | Teste garante que `01`..`06` passam; teto de renda folgado |
| Concept drift fraco demais para derrubar o AUC | Intensidades em `config.py`, calibradas até a queda passar o limiar em `05`/`06` |
| Memória ao ler 307 mil linhas no Docker Desktop | `usecols` + dtypes compactos (`category`, `float32`) |
| Download do Kaggle bloqueado | Caminho manual documentado |

## Histórico de revisões

**2026-10-27 — alinhamento com a implementação.** Decisões tomadas durante a execução, refletidas acima:

- §2 e §3.1: portas do host sobrescrevíveis pelo `.env`; versões fixadas que diferem do plano
  (`pyarrow==17.0.0`, `multimethod==1.10.0`, `sqlalchemy==2.0.36` na imagem do MLflow).
- §3.3: `prepare --synthetic` (base sintética no mesmo schema, usada na demo e no CI).
- §4.2 e §6.1: lotes mensais sorteados de forma independente do pool inteiro, com
  `min(BATCH_SIZE, len(pool) // 6)` linhas; chaves do `manifest.json` (`injected_features` como verdade-base
  e `drifted_features` como subconjunto detectável, mais `expected_rules` no `03-dirty`).
- §5.1 e §5.2: contrato sobre `client_key`, rótulos nativos `column_missing`, `dtype_coercion` e
  `validation_error`; lote bloqueado é copiado, não movido, para a quarentena.
- §6.2: bins do PSI nos pontos médios da grade de 100 percentis da referência (robustos a empates), com o
  alcance da correção; calibração relativa à referência (`scale_pos_weight`); KS hipersensível em variável
  discretizada.
- §6.3, §7.3, §7.4 e §8.3: regra de viés por degradação em relação à referência
  (`FAIRNESS_DEGRADATION_MARGIN = 0,05`), métrica `credit_fairness_disparate_impact_reference`, nova expressão
  do `FairnessDegraded`, `pipeline_stage_status = -1` para bloqueio e `batch_info`; achado permanente de viés
  por idade.
- §8.1: `usecols` com 21 colunas (a contagem de 18 estava errada).
