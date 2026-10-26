# Monitoramento Contínuo de Credit Scoring — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** pipeline batch containerizado que valida lotes mensais de crédito com Pandera, detecta data drift e concept drift com Evidently e PSI/KS, publica métricas, logs e alertas (Pushgateway/Prometheus, Loki, Grafana, MLflow) e documenta a governança LGPD e a análise causal.

**Arquitetura:** pacote Python `credit_monitor` com um CLI (`python -m credit_monitor <comando>`), executado no serviço `pipeline` do `docker compose`, ao lado de `mlflow`, `pushgateway`, `prometheus`, `grafana` e `loki`. Cada etapa do pipeline roda dentro de `stage()`, que registra logs e métricas. Toda integração externa degrada graciosamente: os relatórios em `reports/` são sempre gravados.

**Stack:** Python 3.12, pandas, scikit-learn, xgboost, pandera, evidently 0.7.x, scipy, mlflow 2.17.2, prometheus-client, requests, kaggle, pytest, ruff. Docker Compose v2+, Prometheus, Pushgateway, Grafana, Loki.

**Spec:** `docs/superpowers/specs/2026-10-25-credit-monitoring-design.md` (leia junto com este plano; a seção da spec citada em cada tarefa é a fonte dos valores).

## Restrições globais

- Host Windows sem `make`: todo comando documentado tem forma `docker compose ...`. O `Makefile` é só atalho.
- Comandos de teste: `docker compose run --rm --entrypoint pytest pipeline <args>` (abreviado abaixo como **`PYTEST`**). No CI (Linux), `pytest` direto.
- Comandos do CLI: `docker compose run --rm pipeline <comando>` (abreviado como **`CLI <comando>`**).
- Portas publicadas apenas em `127.0.0.1`.
- `SEED = 42`. Split: referência 60% (treino 70 / validação 15 / teste 15) e pool 40%, estratificado por `TARGET`.
- `CONTRACT_VERSION = "1.0"`. Teto de renda `10_000_000`. Idade entre 18 e 100. Sentinela `DAYS_EMPLOYED = 365243`. Lote mínimo de 1.000 linhas.
- Limiares (spec §6.3): PSI de atenção 0,10 e de drift 0,20; KS D > 0,10; drift share ≥ 0,30; PSI do score ≥ 0,20; queda de AUC > 0,05; `calibration_ratio` > 1,3; impacto disparato < 0,8.
- Lotes: `BATCH_SIZE = 20_000`; ordem `SIMULATION_ORDER = ["01", "02", "03-dirty", "03", "04", "05", "06"]`.
- Nenhum ID bruto (`SK_ID_CURR`) em logs, métricas, MLflow, Loki ou `reports/`. Só `client_key`.
- `CODE_GENDER` nunca é feature do modelo.
- Mensagens de log, relatórios e README em português. Identificadores de código em inglês.
- Commits no formato Conventional Commits, terminando com a linha `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Refinamentos da spec decididos neste plano

1. **Lote bloqueado é copiado, não movido, para `data/quarantine/`.** Assim o `simulate-all` pode rodar de novo.
2. **`pipeline_stage_status` usa três valores:** `1` = sucesso, `0` = erro (exceção), `-1` = bloqueado pelo contrato. O alerta `PipelineStageFailed` (`== 0`) não duplica o `DataContractViolation`.
3. **O label `batch` vem só da *grouping key* do Pushgateway.** As métricas não o declaram. O lote real do grupo `latest` é exposto em `credit_batch_info{batch_id=...} = 1`.
4. **O contrato valida `client_key`**, produzido na ingestão, para unicidade e não nulo. O pool de produção e os lotes guardam `SK_ID_CURR`, porque representam o sistema de origem. A referência já é gravada pseudonimizada.
5. **`prepare --synthetic`** gera uma base sintética com o mesmo schema. O sistema inteiro roda e é verificado antes de haver dataset real, e o CI usa a mesma base.
6. **Ponto de corte de aprovação:** o threshold que maximiza KS (TPR − FPR) na validação. "TPR" na auditoria de viés significa a taxa de aprovação entre bons pagadores (`TARGET = 0`).

## Pontos de atenção na revisão

1. **Reexecutar um lote bloqueado** (`monitor --batch 03-dirty` duas vezes, ou `simulate-all` duas vezes) deve bloquear de novo, sem `FileNotFoundError`. O teste fica na Tarefa 10.
2. **`monitor` antes de `train`, ou com lote inexistente**, deve dar mensagem clara em português e código de saída 2, sem stack trace. Teste na Tarefa 10.
3. **PSI com distribuição degenerada** (referência constante, coluna toda nula no lote, categoria nova) deve devolver um número finito, sem divisão por zero. Teste na Tarefa 9.
4. **Lote sem `TARGET`** deve pular as métricas com rótulo, e a recomendação passa a usar só os sinais sem rótulo. Teste na Tarefa 9.
5. **Serviço externo pendurado** (Loki, MLflow ou Pushgateway num endereço sem resposta) não pode travar o pipeline: timeout de 2 s, desativação após a primeira falha, e o lote completa em menos de 10 s com a infraestrutura fora. Teste na Tarefa 3.

---

## Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `Makefile` | Build e execução |
| `src/credit_monitor/__main__.py`, `cli.py` | Parser e despacho dos comandos |
| `src/credit_monitor/config.py` | `Settings`, colunas, features, limiares, hiperparâmetros, `DRIFT_SCHEDULE` |
| `src/credit_monitor/data/synthetic.py` | Base sintética com schema Home Credit |
| `src/credit_monitor/data/download.py` | Download Kaggle ou checagem manual |
| `src/credit_monitor/data/prepare.py` | Carga mínima, limpeza via contrato, split |
| `src/credit_monitor/data/simulate.py` | Drift, lote sujo, manifests |
| `src/credit_monitor/privacy/pseudonymize.py`, `retention.py` | HMAC do ID e purge |
| `src/credit_monitor/contracts/schema.py`, `validate.py` | Schema Pandera, bloqueio e relatório |
| `src/credit_monitor/model/features.py`, `train.py`, `predict.py`, `fairness.py` | Features, modelo, scoring, viés |
| `src/credit_monitor/drift/stats.py`, `detect.py`, `causal.py` | PSI/KS, Evidently, desempenho, recomendação, causalidade |
| `src/credit_monitor/observability/logging.py`, `metrics.py`, `tracking.py`, `stage.py`, `report_index.py` | Logs JSON + Loki, Pushgateway, MLflow, instrumentação, `index.html` |
| `src/credit_monitor/pipeline.py` | `monitor`, `simulate-all`, `reset-metrics` |
| `infra/mlflow/Dockerfile`, `infra/prometheus/*.yml`, `infra/loki/loki-config.yml`, `infra/grafana/**` | Infraestrutura |
| `.github/workflows/ci.yml` | ruff, pytest, promtool |
| `README.md`, `docs/roteiro-video.md`, `docs/evidence/` | Entregáveis de documentação |

---

### Tarefa 1: Esqueleto do projeto, container e CI

**Arquivos:**
- Criar: `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `Makefile`, `src/credit_monitor/__init__.py`, `src/credit_monitor/__main__.py`, `src/credit_monitor/cli.py`, `src/credit_monitor/config.py`, `tests/__init__.py` (vazio: os testes importam helpers uns dos outros, como `from tests.test_observability import _settings`), `tests/test_cli.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produz:
  - `config.Settings` (dataclass congelada), com `Settings.from_env() -> Settings`. Campos: `data_dir: Path` (`/app/data`), `reports_dir: Path` (`/app/reports`), `models_dir: Path` (`/app/models`), `mlflow_uri: str | None`, `pushgateway_url: str | None`, `loki_url: str | None`, `pseudonymization_salt: str`, `http_timeout: float = 2.0`. As variáveis de ambiente correspondentes são `DATA_DIR`, `REPORTS_DIR`, `MODELS_DIR`, `MLFLOW_TRACKING_URI`, `PUSHGATEWAY_URL`, `LOKI_URL`, `PSEUDONYMIZATION_SALT`.
  - `cli.build_parser() -> argparse.ArgumentParser`, `cli.main(argv: list[str] | None = None) -> int` e o registro `cli.COMMANDS: dict[str, Callable[[Settings, argparse.Namespace], int]]`. As tarefas seguintes adicionam subcomandos aqui.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_cli.py
from credit_monitor.cli import build_parser, main

def test_help_lists_program():
    assert "credit_monitor" in build_parser().format_help()

def test_unknown_command_returns_2(capsys):
    assert main(["nao-existe"]) == 2

def test_settings_from_env(monkeypatch, tmp_path):
    from credit_monitor.config import Settings
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "s")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    s = Settings.from_env()
    assert s.data_dir == tmp_path and s.mlflow_uri is None
```

- [ ] **Passo 2: Criar os arquivos de build**
  - `pyproject.toml`: pacote em `src/`, `requires-python = ">=3.12"`. Dependências com versão fixa: `pandas==2.2.3`, `scikit-learn==1.5.2`, `xgboost==2.1.3`, `pandera==0.21.0`, `evidently` (o patch exato da série 0.7 que o pip resolver, gravado como `==`), `scipy==1.14.1`, `pyarrow==18.0.0`, `mlflow==2.17.2`, `prometheus-client==0.21.0`, `requests==2.32.3`, `kaggle==1.6.17`, `joblib==1.4.2`. Extras `dev`: `pytest==8.3.3`, `ruff==0.7.4`. Configuração do ruff com `line-length = 100`.
  - `Dockerfile`: `python:3.12-slim`, `apt-get install libgomp1` (necessário para o xgboost), `pip install -e .[dev]`, `WORKDIR /app`, `ENTRYPOINT ["python", "-m", "credit_monitor"]`.
  - `docker-compose.yml`: só o serviço `pipeline` nesta tarefa (build `.`, `env_file: .env`, bind mounts de `./src`, `./tests`, `./data`, `./reports`, `./models`, `./infra` em `/app/...`).
  - `.env.example`: todas as variáveis do `Settings`, mais `KAGGLE_USERNAME`, `KAGGLE_KEY` e `GF_SECURITY_ADMIN_PASSWORD`. URLs apontando para os nomes de serviço (`http://mlflow:5000`, `http://pushgateway:9091`, `http://loki:3100`).
  - `.gitignore`: `data/`, `reports/`, `models/`, `.env`, `__pycache__/`, `.pytest_cache/`, `mlruns/`.
  - `Makefile`: alvos `build up down test prepare train simulate causal`, cada um equivalente ao comando `docker compose` correspondente.

- [ ] **Passo 3: Implementar `config.py` (só o `Settings`) e `cli.py`**
  `main` retorna 2 em caso de comando desconhecido (capturar o `SystemExit` do argparse) e despacha para `COMMANDS[args.command](settings, args)`. `__main__.py` faz `raise SystemExit(main())`.

- [ ] **Passo 4: Build e testes**
  Rodar: `copy .env.example .env`, depois `docker compose build`, depois `PYTEST tests/test_cli.py -v`.
  Esperado: 3 testes passando.

- [ ] **Passo 5: Criar o CI** (`.github/workflows/ci.yml`)
  Job `test` em `ubuntu-latest`, Python 3.12: `pip install -e .[dev]`, `ruff check .`, `pytest -q`, com `PSEUDONYMIZATION_SALT=ci`. O job `promtool` entra na Tarefa 11.

- [ ] **Passo 6: Commit**
```bash
git add -A && git commit -m "build: scaffold credit_monitor package, container and CI"
```

---

### Tarefa 2: Configuração de dados e base sintética

**Arquivos:**
- Modificar: `src/credit_monitor/config.py`
- Criar: `src/credit_monitor/data/__init__.py`, `src/credit_monitor/data/synthetic.py`, `tests/conftest.py`, `tests/test_synthetic.py`

**Interfaces:**
- Produz, em `config`:
  - `RAW_COLUMNS: list[str]`: as 22 colunas da spec §4.1 (inclui `SK_ID_CURR`, `CODE_GENDER`, `TARGET`).
  - `RAW_DTYPES: dict[str, str]`: categóricas como `"category"`, valores contínuos como `"float32"`.
  - `CATEGORICAL_FEATURES: list[str]`: as 7 colunas `NAME_*`/`FLAG_OWN_*`.
  - `NUMERIC_FEATURES: list[str]`: `AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, AGE_YEARS, YEARS_EMPLOYED, IS_UNEMPLOYED_OR_PENSIONER, CNT_CHILDREN, CNT_FAM_MEMBERS, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM`.
  - `MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES` (22 itens).
  - Constantes: `SEED`, `DAYS_EMPLOYED_SENTINEL = 365243`, `INCOME_MAX`, `AGE_MIN`, `AGE_MAX`, `MIN_BATCH_ROWS`, `BATCH_SIZE`, `SIMULATION_ORDER` e os limiares das Restrições globais (`PSI_WARN`, `PSI_DRIFT`, `KS_DRIFT`, `DRIFT_SHARE_ALERT`, `PRED_PSI_ALERT`, `AUC_DROP_ALERT`, `CALIBRATION_ALERT`, `DISPARATE_IMPACT_MIN`).
- Produz `synthetic.make_synthetic_application(n: int, seed: int = SEED, with_anomalies: bool = True) -> pd.DataFrame`, com as colunas `RAW_COLUMNS`:
  - `SK_ID_CURR = 900_000_001 + i`, um intervalo distinto para os testes de vazamento acharem os IDs.
  - Domínios categóricos iguais aos do Home Credit (por exemplo `NAME_INCOME_TYPE` inclui `"Working"`, `"Pensioner"`, `"Unemployed"`).
  - Cerca de 18% com `DAYS_EMPLOYED = 365243`.
  - `EXT_SOURCE_1` com cerca de 55% de nulos.
  - `TARGET` amostrado de uma logística sobre `EXT_SOURCE_*`, `ANNUITY_INCOME_RATIO`, idade e desemprego, com taxa base de cerca de 8%.
  - Se `with_anomalies=True`, injeta exatamente, **em 17 linhas distintas**: 1 renda de `117_000_000`, 4 `CODE_GENDER = "XNA"` e 12 `AMT_ANNUITY` nulos.
- Produz o fixture `synthetic_raw` (n = 30.000, escopo de sessão) em `tests/conftest.py`.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_synthetic.py
from credit_monitor import config
from credit_monitor.data.synthetic import make_synthetic_application

def test_schema_and_anomalies():
    df = make_synthetic_application(5_000, seed=1)
    assert list(df.columns) == config.RAW_COLUMNS
    assert (df["AMT_INCOME_TOTAL"] == 117_000_000).sum() == 1
    assert (df["CODE_GENDER"] == "XNA").sum() == 4
    assert df["AMT_ANNUITY"].isna().sum() == 12
    assert df["SK_ID_CURR"].min() >= 900_000_001

def test_target_rate_and_determinism():
    a = make_synthetic_application(20_000, seed=7)
    b = make_synthetic_application(20_000, seed=7)
    assert 0.06 <= a["TARGET"].mean() <= 0.10
    assert a.equals(b)

def test_model_features_count():
    assert len(config.MODEL_FEATURES) == 22
    assert "CODE_GENDER" not in config.MODEL_FEATURES
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_synthetic.py -v`. Esperado: `ModuleNotFoundError`.
- [ ] **Passo 3: Implementar as constantes em `config.py` e o `make_synthetic_application`.**
- [ ] **Passo 4: Rodar e ver passar.** Mesmo comando. Esperado: 3 testes passando.
- [ ] **Passo 5: Commit.** `feat(data): add data config and synthetic Home Credit generator`

---

### Tarefa 3: Núcleo de observabilidade (logs, Loki, métricas, MLflow, `stage`)

**Arquivos:**
- Criar: `src/credit_monitor/observability/{__init__,logging,metrics,tracking,stage}.py`, `tests/test_observability.py`

**Interfaces:**
- `logging.get_logger(settings: Settings, run_id: str) -> logging.LoggerAdapter`
  - Formatter JSON com os campos `ts, level, run_id, batch_id, stage, event, status, duration_ms, rows_in, rows_out, message`. Campos ausentes viram `null`; os extras chegam via `extra=`.
  - Handlers: stdout; arquivo `reports/logs/pipeline.jsonl`; e `LokiHandler` se `settings.loki_url` existir.
- `logging.LokiHandler(url: str, timeout: float)`
  - Faz POST síncrono em `/loki/api/v1/push` com labels `{job="credit_pipeline", stage, level, batch}`.
  - Na primeira exceção, define `self.disabled = True` e emite um único aviso no stderr.
- `metrics.METRIC_CATALOG: dict[str, tuple[str, list[str]]]`
  - Mapeia o nome completo (com prefixo `credit_`) para a tupla (descrição em português, labels **sem** `batch`).
  - Contém todas as métricas da spec §7.3, mais `credit_batch_info` (`["batch_id"]`).
  - Exemplo: `"credit_feature_psi": ("PSI da feature vs referência", ["feature"])`.
- `metrics.MetricsPublisher(settings: Settings, batch_id: str)`, com os métodos:
  - `.set(name: str, value: float, **labels)`: levanta `KeyError` se o nome não estiver no catálogo.
  - `.push() -> bool`: `push_to_gateway` (PUT) duas vezes, com `grouping_key={"batch": batch_id}` e `{"batch": "latest"}`, mais `credit_batch_info{batch_id}=1`. Retorna `False` e loga `WARN` se o gateway faltar ou falhar.
  - `.values() -> dict`: snapshot para o MLflow e os testes.
- `tracking.Tracker(settings: Settings)`
  - Na construção, faz `GET {uri}/health` com timeout. Se falhar, `self.enabled = False`.
  - `@contextmanager run(experiment: str, run_name: str, tags: dict[str, str])`, `log_metrics(d: dict[str, float])`, `log_params(d)`, `log_artifact(path: Path)`, `log_model(...)` (Tarefa 8), `active_run_id: str | None`.
  - Desabilitado, tudo vira no-op.
- `stage.stage(name: str, batch_id: str, publisher: MetricsPublisher, logger) -> ContextManager[StageContext]`
  - `StageContext` expõe os atributos `rows_in`, `rows_out` e o método `block(reason: str)`.
  - Na saída, publica `credit_pipeline_stage_status{stage}` (1/0/-1), `credit_pipeline_stage_duration_seconds{stage}` e `credit_pipeline_rows_processed{stage}` (= `rows_out`), e loga `event="stage_end"`.
  - Em caso de exceção, status 0, log `ERROR` e a exceção é relançada.
- `stage.peak_memory_bytes() -> int`: usa `resource.getrusage`, multiplicando por 1024 no Linux. Retorna 0 se o módulo não existir.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_observability.py
import json, time, pytest
from credit_monitor.observability.metrics import MetricsPublisher, METRIC_CATALOG
from credit_monitor.observability.stage import stage
from credit_monitor.observability.logging import get_logger
from credit_monitor.observability.tracking import Tracker

DEAD = "http://10.255.255.1:9"  # não roteável: força timeout

def _settings(tmp_path, **kw):
    from credit_monitor.config import Settings
    base = dict(data_dir=tmp_path, reports_dir=tmp_path / "reports", models_dir=tmp_path,
                mlflow_uri=None, pushgateway_url=None, loki_url=None,
                pseudonymization_salt="t", http_timeout=2.0)
    return Settings(**{**base, **kw})

def test_stage_success_records_status_and_log(tmp_path):
    s = _settings(tmp_path); pub = MetricsPublisher(s, "01"); log = get_logger(s, "r1")
    with stage("contract", "01", pub, log) as ctx:
        ctx.rows_in = ctx.rows_out = 100
    v = pub.values()
    assert v[("credit_pipeline_stage_status", (("stage", "contract"),))] == 1
    line = json.loads((tmp_path / "reports/logs/pipeline.jsonl").read_text().splitlines()[-1])
    assert line["stage"] == "contract" and line["rows_out"] == 100 and line["batch_id"] == "01"

def test_stage_block_and_error(tmp_path):
    s = _settings(tmp_path); pub = MetricsPublisher(s, "01"); log = get_logger(s, "r1")
    with stage("contract", "01", pub, log) as ctx:
        ctx.block("contrato violado")
    with pytest.raises(ValueError):
        with stage("predict", "01", pub, log):
            raise ValueError("x")
    v = pub.values()
    assert v[("credit_pipeline_stage_status", (("stage", "contract"),))] == -1
    assert v[("credit_pipeline_stage_status", (("stage", "predict"),))] == 0

def test_unknown_metric_rejected(tmp_path):
    with pytest.raises(KeyError):
        MetricsPublisher(_settings(tmp_path), "01").set("credit_nao_existe", 1)

def test_dead_infra_does_not_hang(tmp_path):
    s = _settings(tmp_path, mlflow_uri=DEAD, pushgateway_url=DEAD, loki_url=DEAD)
    t0 = time.monotonic()
    log = get_logger(s, "r1"); pub = MetricsPublisher(s, "01")
    for _ in range(20):
        log.info("evento", extra={"batch_id": "01"})
    assert pub.push() is False and Tracker(s).enabled is False
    assert time.monotonic() - t0 < 10

def test_catalog_has_no_batch_label():
    assert all("batch" not in labels for _, labels in METRIC_CATALOG.values())
```

O formato da chave de `values()` (tupla do nome + tupla ordenada de labels) é decisão desta tarefa e precisa ser mantido nas próximas.

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_observability.py -v`.
- [ ] **Passo 3: Implementar os quatro módulos.**
  O `MetricsPublisher` cria um `CollectorRegistry` próprio por instância e um `Gauge` sob demanda por nome do catálogo.
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 5 testes passando, e o teste da infraestrutura morta em menos de 10 s.
- [ ] **Passo 5: Commit.** `feat(observability): add JSON/Loki logging, Pushgateway publisher, MLflow tracker and stage instrumentation`

---

### Tarefa 4: Privacidade (pseudonimização e retenção)

**Arquivos:**
- Criar: `src/credit_monitor/privacy/{__init__,pseudonymize,retention}.py`, `tests/test_privacy.py`
- Modificar: `src/credit_monitor/config.py` (adicionar `RETENTION_DAYS = {"quarantine": 30, "logs": 90}`), `src/credit_monitor/cli.py` (comando `purge`)

**Interfaces:**
- `pseudonymize.client_key(raw_id, salt: str) -> str | None`
  - HMAC-SHA256 hexadecimal de `str(int(raw_id))`, truncado em 16 caracteres.
  - Retorna `None` se o ID for nulo.
- `pseudonymize.pseudonymize(df: pd.DataFrame, salt: str) -> pd.DataFrame`
  - Devolve uma cópia com a coluna `client_key` no lugar de `SK_ID_CURR`, como **primeira** coluna.
- `retention.purge(settings: Settings, now: datetime | None = None) -> dict[str, int]`
  - Remove arquivos de `data/quarantine/` com mtime anterior a 30 dias e de `reports/logs/` anterior a 90 dias.
  - Retorna a contagem de removidos por área.
- Comando `CLI purge`: imprime as contagens e retorna 0.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_privacy.py
import os, time
import pandas as pd
from datetime import datetime, timedelta
from credit_monitor.privacy.pseudonymize import client_key, pseudonymize

def test_deterministic_and_salted():
    assert client_key(100002, "a") == client_key(100002.0, "a")
    assert client_key(100002, "a") != client_key(100002, "b")
    assert len(client_key(1, "a")) == 16 and client_key(None, "a") is None

def test_pseudonymize_drops_raw_id():
    out = pseudonymize(pd.DataFrame({"SK_ID_CURR": [1, 1, None], "X": [1, 2, 3]}), "s")
    assert list(out.columns) == ["client_key", "X"]
    assert out["client_key"].iloc[0] == out["client_key"].iloc[1]
    assert out["client_key"].isna().iloc[2]

def test_purge_respects_windows(tmp_path):
    from credit_monitor.privacy.retention import purge
    from tests.test_observability import _settings
    q = tmp_path / "quarantine"; q.mkdir(parents=True)
    logs = tmp_path / "reports/logs"; logs.mkdir(parents=True)
    old_q, new_q, old_l = q / "a.parquet", q / "b.parquet", logs / "x.jsonl"
    for p in (old_q, new_q, old_l): p.write_text("x")
    old = time.time() - 100 * 86400
    os.utime(old_q, (old, old)); os.utime(old_l, (old, old))
    assert purge(_settings(tmp_path)) == {"quarantine": 1, "logs": 1}
    assert new_q.exists() and not old_q.exists()
```

(O `data_dir` dos testes é `tmp_path`, então a quarentena fica em `tmp_path/"quarantine"`. O `retention` usa `settings.data_dir / "quarantine"`.)

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_privacy.py -v`.
- [ ] **Passo 3: Implementar `client_key`, `pseudonymize`, `purge` e registrar o `purge` no CLI.**
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 3 testes passando.
- [ ] **Passo 5: Commit.** `feat(privacy): add HMAC pseudonymization and retention purge`

---

### Tarefa 5: Contrato de dados Pandera e bloqueio

**Arquivos:**
- Criar: `src/credit_monitor/contracts/{__init__,schema,validate}.py`, `tests/test_contracts.py`

**Interfaces:**
- `schema.CONTRACT_VERSION = "1.0"`.
- `schema.build_schema(categorical_domains: dict[str, list[str]]) -> pa.DataFrameSchema`. Regras da spec §5.1 aplicadas ao DataFrame **pseudonimizado** (`client_key` no lugar de `SK_ID_CURR`). Cada `Check` recebe `name=` estável, porque esse nome é o label `rule` das métricas:
  - `client_key_unique`, `client_key_not_null`;
  - `income_not_null`, `income_range`;
  - `age_range` (sobre `DAYS_BIRTH`: entre `-AGE_MAX*365.25` e `-AGE_MIN*365.25`);
  - `days_employed_valid`;
  - `credit_positive`, `annuity_not_null`, `annuity_positive`;
  - `ext_source_range` (as três colunas, aceitando nulos);
  - `domain_<COLUNA>` para cada categórica e para `CODE_GENDER ∈ {M, F}`;
  - `target_binary` (coluna `TARGET` com `required=False`);
  - `min_rows` (check de DataFrame, mínimo `MIN_BATCH_ROWS`).
  - Colunas obrigatórias = `RAW_COLUMNS` sem `SK_ID_CURR` e sem `TARGET`, mais `client_key`. Os dtypes são coagidos (`coerce=True`).
- `schema.domains_from_reference(df: pd.DataFrame) -> dict[str, list[str]]`: categorias observadas na referência limpa. `CODE_GENDER` é fixado em `["F", "M"]`.
- `validate.ContractResult` (dataclass): `passed: bool`, `n_rows: int`, `violations: dict[str, int]` (regra → contagem), `failures: pd.DataFrame` (colunas `rule, column, failure_value, client_key`), `failing_index: pd.Index`.
- `validate.validate_batch(df: pd.DataFrame, domains: dict) -> ContractResult`: usa `lazy=True` e nunca levanta erro de schema.
- `validate.write_failure_report(result: ContractResult, batch_id: str, out_dir: Path) -> tuple[Path, Path]`: gera `<batch>_failures.json` (com `contract_version`, `violations` e até 500 linhas de `failures`) e `<batch>_failures.html` (tabela de regra → contagem, mais amostra).
- `validate.quarantine(src: Path, batch_id: str, settings: Settings) -> Path`: **copia** para `data/quarantine/<batch>.parquet`.

- [ ] **Passo 1: Escrever o teste que falha.** Use o fixture `synthetic_raw`, limpo das 17 anomalias e pseudonimizado, como `valid_df`.

```python
# tests/test_contracts.py
import pytest
from credit_monitor.contracts.validate import validate_batch

@pytest.fixture(scope="module")
def valid(synthetic_raw):
    from credit_monitor.privacy.pseudonymize import pseudonymize
    from credit_monitor.contracts.schema import domains_from_reference
    df = synthetic_raw[(synthetic_raw.AMT_INCOME_TOTAL < 1e7) & (synthetic_raw.CODE_GENDER != "XNA")
                       & synthetic_raw.AMT_ANNUITY.notna()]
    df = pseudonymize(df, "t").reset_index(drop=True)
    return df, domains_from_reference(df)

def test_valid_passes(valid):
    df, dom = valid
    assert validate_batch(df, dom).passed

@pytest.mark.parametrize("mutate,rule", [
    (lambda d: d.assign(AMT_INCOME_TOTAL=d.AMT_INCOME_TOTAL.where(d.index != 0)), "income_not_null"),
    (lambda d: d.assign(AMT_INCOME_TOTAL=d.AMT_INCOME_TOTAL.where(d.index != 0, 2e7)), "income_range"),
    (lambda d: d.assign(DAYS_BIRTH=d.DAYS_BIRTH.where(d.index != 0, -15 * 365.25)), "age_range"),
    (lambda d: d.assign(DAYS_EMPLOYED=d.DAYS_EMPLOYED.where(d.index != 0, 500)), "days_employed_valid"),
    (lambda d: d.assign(AMT_CREDIT=d.AMT_CREDIT.where(d.index != 0, -1)), "credit_positive"),
    (lambda d: d.assign(EXT_SOURCE_2=d.EXT_SOURCE_2.where(d.index != 0, 1.5)), "ext_source_range"),
    (lambda d: d.assign(client_key=d.client_key.where(d.index != 1, d.client_key[0])), "client_key_unique"),
    (lambda d: d.astype({"NAME_CONTRACT_TYPE": object}).assign(
        NAME_CONTRACT_TYPE=lambda x: x.NAME_CONTRACT_TYPE.where(x.index != 0, "Crypto")),
     "domain_NAME_CONTRACT_TYPE"),
    (lambda d: d.head(500), "min_rows"),
])
def test_each_rule_fails(valid, mutate, rule):
    df, dom = valid
    r = validate_batch(mutate(df.copy()), dom)
    assert not r.passed and r.violations.get(rule, 0) >= 1

def test_failures_never_contain_raw_id(valid):
    df, dom = valid
    r = validate_batch(df.assign(AMT_CREDIT=-1.0), dom)
    assert "SK_ID_CURR" not in r.failures.columns
    assert set(r.failures.columns) == {"rule", "column", "failure_value", "client_key"}
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_contracts.py -v`.
- [ ] **Passo 3: Implementar `schema.py` e `validate.py`.**
  Os `failure_cases` do `SchemaErrors` são mapeados para `rule` (o nome do check), `column`, `failure_value` e `client_key`, este último buscado pelo índice da linha (nulo nas falhas de DataFrame).
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 11 testes passando.
- [ ] **Passo 5: Commit.** `feat(contracts): add Pandera data contract with blocking validation and failure reports`

---

### Tarefa 6: Download e preparação da referência

**Arquivos:**
- Criar: `src/credit_monitor/data/download.py`, `src/credit_monitor/data/prepare.py`, `tests/test_prepare.py`
- Modificar: `src/credit_monitor/cli.py` (comandos `download` e `prepare [--synthetic] [--n 60000]`)

**Interfaces:**
- `download.ensure_dataset(settings: Settings) -> Path`
  - Se `data/raw/application_train.csv` existe, retorna o caminho.
  - Senão, se `KAGGLE_USERNAME` e `KAGGLE_KEY` existem, roda `kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw` e descompacta.
  - Senão, levanta `DatasetUnavailable`, com mensagem em português explicando os dois caminhos (incluindo aceitar as regras da competição).
- `prepare.load_raw(path: Path) -> pd.DataFrame`: `pd.read_csv(usecols=RAW_COLUMNS, dtype=RAW_DTYPES)`.
- `prepare.clean_reference(raw: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, ContractResult]`
  1. Pseudonimiza uma cópia.
  2. Calcula `domains_from_reference` excluindo `CODE_GENDER = "XNA"`.
  3. Valida.
  4. Remove `failing_index` do bruto e devolve o bruto limpo (ainda com `SK_ID_CURR`) e o resultado da validação original.
- `prepare.split_reference(clean: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]`: chaves `train`, `valid`, `test` e `pool`, estratificado por `TARGET`.
- `prepare.run_prepare(settings: Settings, synthetic: bool, n: int) -> dict[str, Path]`. Grava:
  - `data/reference/{train,valid,test}.parquet`, **pseudonimizados**;
  - `data/reference/production_pool.parquet`, com `SK_ID_CURR`;
  - `data/reference/domains.json`;
  - `reports/contracts/raw_reference_failures.json` e `.html`.
  - Nesta tarefa o `run_prepare` só grava a referência. A Tarefa 7 acrescenta a chamada a `simulate.build_all_batches`.
- O comando `prepare` sem `--synthetic` chama `ensure_dataset` e, se der `DatasetUnavailable`, imprime a mensagem e retorna 2.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_prepare.py
import json
from credit_monitor.data.prepare import clean_reference, split_reference

def test_clean_reference_catches_real_anomalies(synthetic_raw):
    clean, result = clean_reference(synthetic_raw, "t")
    assert not result.passed
    assert result.violations["income_range"] == 1
    assert result.violations["domain_CODE_GENDER"] == 4
    assert result.violations["annuity_not_null"] == 12
    assert len(clean) == len(synthetic_raw) - 17

def test_split_proportions_and_stratification(synthetic_raw):
    clean, _ = clean_reference(synthetic_raw, "t")
    parts = split_reference(clean, 42)
    n = len(clean)
    assert abs(len(parts["pool"]) / n - 0.40) < 0.01
    assert abs(len(parts["train"]) / n - 0.42) < 0.01
    rates = [p["TARGET"].mean() for p in parts.values()]
    assert max(rates) - min(rates) < 0.01

def test_run_prepare_synthetic_writes_pseudonymized_reference(tmp_path):
    from credit_monitor.data.prepare import run_prepare
    from tests.test_observability import _settings
    import pandas as pd
    s = _settings(tmp_path)
    run_prepare(s, synthetic=True, n=20_000)
    train = pd.read_parquet(tmp_path / "reference/train.parquet")
    pool = pd.read_parquet(tmp_path / "reference/production_pool.parquet")
    assert "SK_ID_CURR" not in train.columns and "client_key" in train.columns
    assert "SK_ID_CURR" in pool.columns
    assert json.loads((tmp_path / "reports/contracts/raw_reference_failures.json").read_text())["violations"]

def test_missing_dataset_message(tmp_path, monkeypatch):
    from credit_monitor.data.download import ensure_dataset, DatasetUnavailable
    from tests.test_observability import _settings
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    import pytest
    with pytest.raises(DatasetUnavailable, match="application_train.csv"):
        ensure_dataset(_settings(tmp_path))
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_prepare.py -v`.
- [ ] **Passo 3: Implementar `download.py`, `prepare.py` e os comandos do CLI.**
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 4 testes passando.
- [ ] **Passo 5: Smoke test pelo CLI.** `CLI prepare --synthetic`. Esperado: arquivos em `data/reference/` e mensagem com o número de linhas removidas.
- [ ] **Passo 6: Commit.** `feat(data): add Kaggle download and reference preparation with contract-based cleaning`

---

### Tarefa 7: Simulação de drift e lotes mensais

**Arquivos:**
- Criar: `src/credit_monitor/data/simulate.py`, `tests/test_simulate.py`
- Modificar: `src/credit_monitor/config.py` (`DriftParams`, `DRIFT_SCHEDULE`), `src/credit_monitor/data/prepare.py` (o `run_prepare` passa a chamar `build_all_batches`)

**Interfaces:**
- `config.DriftParams`: dataclass congelada com os campos `income_infl, credit_infl, annuity_infl, unemployed_target (float | None), employed_shrink, young_weight, ext1_extra_null, ext23_shrink, concept_flip`.
- `config.DRIFT_SCHEDULE: dict[str, DriftParams]`. Valores da spec §6.1, completados assim:

| Lote | income / credit / annuity | unemployed_target | employed_shrink | young_weight | ext1_extra_null | ext23_shrink | concept_flip |
|---|---|---|---|---|---|---|---|
| 01 | 0 / 0 / 0 | None | 0 | 0 | 0 | 0 | 0 |
| 02 | .02 / .04 / .06 | None | 0 | 0 | 0 | 0 | 0 |
| 03 | .05 / .10 / .14 | .22 | .05 | 0 | 0 | 0 | 0 |
| 04 | .08 / .16 / .22 | .25 | .10 | 1.0 | .10 | .05 | .10 |
| 05 | .10 / .20 / .28 | .28 | .15 | 2.0 | .20 | .10 | .18 |
| 06 | .12 / .25 / .35 | .30 | .20 | 3.0 | .30 | .15 | .25 |

`credit_infl` vale também para `AMT_GOODS_PRICE`.

- `simulate.sample_batch(pool: pd.DataFrame, available: np.ndarray, size: int, young_weight: float, rng) -> np.ndarray`: amostragem sem reposição, com peso `1 + young_weight * (idade < 30)`.
- `simulate.apply_drift(df: pd.DataFrame, p: DriftParams, annuity_ratio_p75: float, rng) -> tuple[pd.DataFrame, pd.Series]`. Retorna o lote alterado e os `TARGET` originais (indexados por `SK_ID_CURR`). Regras:
  - multiplicar renda, crédito, preço do bem e anuidade pelos fatores `(1 + infl)`;
  - converter empregados aleatórios em `DAYS_EMPLOYED = 365243` e `NAME_INCOME_TYPE = "Unemployed"` até a fração de sentinela atingir `unemployed_target`;
  - multiplicar o `DAYS_EMPLOYED` dos empregados restantes por `(1 - employed_shrink)`;
  - anular uma fração `ext1_extra_null` dos `EXT_SOURCE_1` não nulos;
  - multiplicar `EXT_SOURCE_2/3` por `(1 - ext23_shrink)` com clip em `[0, 1]`;
  - concept drift: segmento = `AMT_ANNUITY / AMT_INCOME_TOTAL > annuity_ratio_p75` ou sentinela; virar `concept_flip` dos `TARGET == 0` do segmento para 1.
- `simulate.make_dirty(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, list[str]]`. Injeta 5 linhas de cada defeito e retorna a lista de regras esperadas: `["income_not_null", "age_range", "client_key_unique", "credit_positive", "domain_NAME_CONTRACT_TYPE", "ext_source_range"]`.
- `simulate.build_all_batches(settings: Settings, pool: pd.DataFrame, reference_train: pd.DataFrame, seed: int) -> list[str]`. Gera em `data/batches/<id>/` os arquivos:
  - `batch.parquet` (com `SK_ID_CURR`, `CODE_GENDER` e `TARGET`);
  - `manifest.json`: `batch_id`, `params` (asdict), `drifted_features` (lista das `MODEL_FEATURES` afetadas pelos parâmetros não nulos: inflação → `AMT_*` e as razões; desemprego → `YEARS_EMPLOYED`, `IS_UNEMPLOYED_OR_PENSIONER`, `NAME_INCOME_TYPE`; `young_weight` → `AGE_YEARS`; ext → `EXT_SOURCE_*`), `n_rows`, `expected_rules` (só no lote sujo);
  - `labels_original.parquet`.
  - `03-dirty` é derivado do `03`. Retorna os IDs gerados.

- [ ] **Passo 1: Escrever o teste que falha.**
  O fixture `prepared` fica em `tests/conftest.py`, com escopo de **sessão**, porque as Tarefas 9, 10 e 12 o reutilizam. Ele roda `run_prepare(synthetic=True, n=60_000)` e devolve o `Settings`, com `models_dir = <tmp>/models` separado do `data_dir`.

```python
# tests/conftest.py (acréscimo)
@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    from credit_monitor.data.prepare import run_prepare
    from tests.test_observability import _settings
    root = tmp_path_factory.mktemp("sim")
    s = _settings(root, models_dir=root / "models"); run_prepare(s, synthetic=True, n=60_000)
    return s
```

```python
# tests/test_simulate.py
import json, pandas as pd
from credit_monitor import config

def _load(s, b):
    from credit_monitor.privacy.pseudonymize import pseudonymize
    return pseudonymize(pd.read_parquet(s.data_dir / f"batches/{b}/batch.parquet"), "t")

def test_all_batches_created(prepared):
    for b in config.SIMULATION_ORDER:
        assert (prepared.data_dir / f"batches/{b}/manifest.json").exists()

def test_clean_batches_pass_contract_and_dirty_fails(prepared):
    from credit_monitor.contracts.validate import validate_batch
    dom = json.loads((prepared.data_dir / "reference/domains.json").read_text())
    for b in ["01", "02", "03", "04", "05", "06"]:
        assert validate_batch(_load(prepared, b), dom).passed, b
    r = validate_batch(_load(prepared, "03-dirty"), dom)
    expected = json.loads((prepared.data_dir / "batches/03-dirty/manifest.json").read_text())["expected_rules"]
    assert set(expected) <= set(r.violations)

def test_drift_is_progressive(prepared):
    inc = [_load(prepared, b)["AMT_INCOME_TOTAL"].median() for b in ["01", "06"]]
    assert inc[1] / inc[0] > 1.08
    unemp = (_load(prepared, "06")["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL).mean()
    assert 0.28 <= unemp <= 0.32

def test_concept_drift_raises_default_and_keeps_original(prepared):
    b06 = pd.read_parquet(prepared.data_dir / "batches/06/batch.parquet")
    orig = pd.read_parquet(prepared.data_dir / "batches/06/labels_original.parquet")
    assert b06["TARGET"].mean() > orig["TARGET"].mean() + 0.02
    assert len(orig) == len(b06)

def test_batch_01_has_no_drifted_features(prepared):
    m = json.loads((prepared.data_dir / "batches/01/manifest.json").read_text())
    assert m["drifted_features"] == []
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_simulate.py -v`.
- [ ] **Passo 3: Implementar `DriftParams`, `DRIFT_SCHEDULE` e `simulate.py`, e ligar no `run_prepare`.**
  O `annuity_ratio_p75` vem do `reference_train`. O RNG é `np.random.default_rng(seed)`, com uma sequência por lote derivada de `seed + índice`.
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 5 testes passando.
- [ ] **Passo 5: Commit.** `feat(drift): simulate progressive monthly data and concept drift with manifests`

---

### Tarefa 8: Features, modelo base e auditoria de viés

**Arquivos:**
- Criar: `src/credit_monitor/model/{__init__,features,train,predict,fairness}.py`, `tests/test_model.py`
- Modificar: `config.py` (adicionar `XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", early_stopping_rounds=50, random_state=SEED)` e `AGE_BANDS = [(0, 30, "<30"), (30, 45, "30-45"), (45, 60, "45-60"), (60, 200, "60+")]`), `cli.py` (comando `train`)

**Interfaces:**
- `features.build_features(df: pd.DataFrame) -> pd.DataFrame`
  - Adiciona `AGE_YEARS = -DAYS_BIRTH/365.25`, `YEARS_EMPLOYED` (NaN se sentinela, senão `-DAYS_EMPLOYED/365.25`), `IS_UNEMPLOYED_OR_PENSIONER` (0/1), `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO` e `CREDIT_TERM`.
  - Mantém as outras colunas.
- `features.model_matrix(df) -> pd.DataFrame`: `build_features(df)[MODEL_FEATURES]`.
- `train.build_pipeline(scale_pos_weight: float) -> sklearn.Pipeline`: `ColumnTransformer` (`OneHotEncoder(handle_unknown="ignore")` nas categóricas; `passthrough` nas numéricas) seguido de `XGBClassifier(**XGB_PARAMS, scale_pos_weight=...)`.
- `train.ks_statistic(y_true, score) -> float` (max TPR − FPR) e `train.choose_cutoff(y_true, score) -> float` (threshold do KS máximo).
- `train.run_train(settings: Settings) -> dict`
  - Lê `data/reference/{train,valid,test}.parquet`.
  - Valida train/valid/test com o contrato. Se falhar, levanta `RuntimeError`.
  - Faz o fit com `eval_set` da validação.
  - Calcula as métricas no teste: `auc, pr_auc, ks, brier`.
  - Grava `models/model.joblib` e `models/metadata.json`. O metadata contém `features`, `cutoff`, `contract_version`, `reference_metrics`, `mlflow_model_version` (ou `null`), `trained_at` e `fairness`.
  - Grava `data/reference/test_scored.parquet` (o teste mais a coluna `score`).
  - `metadata["fairness"]` = `fairness_report(...)` no teste, mais `"proxy_top5"` (as 5 primeiras de `proxy_scores`, como dict).
  - Registra no MLflow: experimento `credit-scoring-training`, modelo `credit-scoring`, alias `production`.
  - Retorna o metadata.
- `predict.load_model(settings) -> tuple[Pipeline, dict]`: levanta `ModelNotTrained` se `model.joblib` não existir.
- `predict.score(pipeline, df) -> np.ndarray`: probabilidade da classe 1 sobre `model_matrix(df)`.
- `fairness.fairness_report(df: pd.DataFrame, approved: np.ndarray, y_true: np.ndarray | None) -> dict`. Formato:
  `{"gender": {"approval_rate": {...}, "disparate_impact": float, "tpr": {...} | None, "eo_difference": float | None}, "age_band": {...}}`.
  - O impacto disparato é a menor taxa de aprovação dividida pela maior, entre grupos.
  - A TPR é a aprovação entre `TARGET = 0`.
- `fairness.proxy_scores(X: pd.DataFrame, gender: pd.Series) -> pd.Series`: `mutual_info_classif` por feature, em ordem decrescente.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_model.py
import numpy as np, pandas as pd, pytest
from credit_monitor import config
from credit_monitor.model.features import build_features, model_matrix
from credit_monitor.model.fairness import fairness_report
from credit_monitor.model.train import ks_statistic

def test_features_handle_sentinel():
    df = pd.DataFrame({"DAYS_BIRTH": [-365.25 * 30], "DAYS_EMPLOYED": [365243],
                       "AMT_INCOME_TOTAL": [100.0], "AMT_CREDIT": [500.0], "AMT_ANNUITY": [50.0]})
    f = build_features(df)
    assert f.AGE_YEARS.iloc[0] == pytest.approx(30)
    assert np.isnan(f.YEARS_EMPLOYED.iloc[0]) and f.IS_UNEMPLOYED_OR_PENSIONER.iloc[0] == 1
    assert f.CREDIT_TERM.iloc[0] == 10 and f.ANNUITY_INCOME_RATIO.iloc[0] == 0.5

def test_ks_perfect_and_random():
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0
    rng = np.random.default_rng(0)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.05

def test_disparate_impact():
    df = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4, "DAYS_BIRTH": [-365.25 * 40] * 8})
    r = fairness_report(df, np.array([1, 1, 0, 0, 1, 1, 1, 1]), None)
    assert r["gender"]["disparate_impact"] == pytest.approx(0.5)

def test_run_train_synthetic(tmp_path):
    from credit_monitor.data.prepare import run_prepare
    from credit_monitor.model.train import run_train
    from tests.test_observability import _settings
    s = _settings(tmp_path); run_prepare(s, synthetic=True, n=30_000)
    meta = run_train(s)
    assert meta["reference_metrics"]["auc"] > 0.65
    assert "CODE_GENDER" not in meta["features"] and 0 < meta["cutoff"] < 1
    assert (tmp_path / "reference/test_scored.parquet").exists()
    assert set(meta["fairness"]) == {"gender", "age_band", "proxy_top5"}
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_model.py -v`.
- [ ] **Passo 3: Implementar os quatro módulos e o comando `train`.**
  O `Tracker` ganha `log_model(pipeline, registered_name: str) -> str | None` (`mlflow.sklearn.log_model` + `MlflowClient().set_registered_model_alias(name, "production", version)`), que retorna a versão ou `None` quando desabilitado.
- [ ] **Passo 4: Rodar e ver passar.** Esperado: 4 testes passando.
- [ ] **Passo 5: Commit.** `feat(model): add feature engineering, XGBoost baseline, cutoff selection and fairness audit`

---

### Tarefa 9: Estatística de drift, Evidently, desempenho e recomendação

**Arquivos:**
- Criar: `src/credit_monitor/drift/{__init__,stats,detect}.py`, `tests/test_drift.py`

**Interfaces:**
- `stats.psi(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-4) -> float`
  - Numérica: bordas pelos quantis da referência (`np.unique`, extremos `-inf/inf`) e **bin próprio para NaN**.
  - Categórica (dtype `object`/`category`): uma categoria por bin, incluindo `"__missing__"` e categorias que só aparecem no lote atual.
  - Proporções com `eps` somado. Sempre finito.
- `stats.ks(reference: pd.Series, current: pd.Series) -> float`: estatística D de `ks_2samp` sobre os valores não nulos. Retorna 0.0 se algum lado estiver vazio.
- `stats.feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame`
  - Colunas `feature, kind, psi, ks, status`.
  - `status` vale `drift` (PSI ≥ 0,20, ou KS > 0,10 nas numéricas), `atencao` (0,10 ≤ PSI < 0,20) ou `ok`.
  - Calculada sobre `model_matrix` dos dois lados.
- `detect.performance(y_true: np.ndarray, score: np.ndarray, cutoff: float) -> dict`: `auc, ks, default_rate_observed, mean_predicted_pd, calibration_ratio, approval_rate` (aprovado se `score < cutoff`).
- `detect.recommend(summary: dict, reference_auc: float) -> str`: aplica a regra da spec §6.3 e retorna `"RETREINAR"`, `"INVESTIGAR"` ou `"OK"`.
- `detect.evidently_reports(ref_scored: pd.DataFrame, cur_scored: pd.DataFrame, batch_id: str, out_dir: Path, labeled: bool) -> dict`
  - Gera `<batch>_data_drift.html/.json` (DataDriftPreset com PSI e limiar 0,2, sobre `MODEL_FEATURES + ["score"]`) e, se `labeled`, `<batch>_performance.html/.json` (ClassificationPreset, target `TARGET`, probabilidade `score`).
  - Retorna `{"paths": {...}, "psi": {feature: valor}}`, com o PSI extraído do resultado do Evidently.
  - **Primeiro subpasso desta implementação:** confirmar a API exata da versão 0.7 instalada (`python -c "import evidently; print(evidently.__version__)"` e a doc correspondente) e isolar todo o uso do Evidently nesta função.
- `detect.analyze_batch(ref_scored: pd.DataFrame, cur_scored: pd.DataFrame, batch_id: str, meta: dict, out_dir: Path) -> dict`
  - Monta o resumo e grava `reports/drift/<batch>_summary.json`.
  - Chaves: `batch_id`, `features` (registros da tabela de drift), `drifted_features`, `dataset_drift_share`, `prediction_psi`, `performance` (ou `null` se faltar `TARGET`), `fairness` (impacto disparato por atributo), `recommendation`, `evidently` (caminhos).
  - `dataset_drift_share` = features com `status == "drift"` / 22.

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_drift.py
import numpy as np, pandas as pd, pytest
from credit_monitor.drift.stats import psi, ks
from credit_monitor.drift.detect import performance, recommend

rng = np.random.default_rng(0)

def test_psi_identical_near_zero_and_shift_large():
    x = pd.Series(rng.normal(size=20_000))
    assert psi(x, x.sample(frac=1, random_state=1)) < 0.01
    assert psi(x, x * 1.3 + 0.5) > 0.2

@pytest.mark.parametrize("ref,cur", [
    (pd.Series([1.0] * 1000), pd.Series(rng.normal(size=1000))),
    (pd.Series(rng.normal(size=1000)), pd.Series([np.nan] * 1000)),
    (pd.Series(["a", "b"] * 500), pd.Series(["c"] * 1000)),
])
def test_psi_degenerate_is_finite(ref, cur):
    assert np.isfinite(psi(ref, cur))

def test_psi_detects_null_rate_change():
    ref = pd.Series(np.where(rng.random(20_000) < 0.5, np.nan, rng.random(20_000)))
    cur = pd.Series(np.where(rng.random(20_000) < 0.8, np.nan, rng.random(20_000)))
    assert psi(ref, cur) > 0.2

def test_ks_empty_side_is_zero():
    assert ks(pd.Series([np.nan] * 10), pd.Series([1.0, 2.0])) == 0.0

def test_performance_and_calibration():
    y = np.array([0] * 90 + [1] * 10); s = np.full(100, 0.05)
    p = performance(y, s, cutoff=0.5)
    assert p["calibration_ratio"] == pytest.approx(2.0) and p["approval_rate"] == 1.0

def test_recommend_rules():
    base = {"dataset_drift_share": 0.0, "prediction_psi": 0.0, "fairness": {"gender": 1.0},
            "performance": {"auc": 0.75, "calibration_ratio": 1.0}}
    assert recommend(base, 0.75) == "OK"
    assert recommend({**base, "dataset_drift_share": 0.4}, 0.75) == "INVESTIGAR"
    assert recommend({**base, "performance": {"auc": 0.69, "calibration_ratio": 1.0}}, 0.75) == "RETREINAR"
    assert recommend({**base, "performance": None, "prediction_psi": 0.3}, 0.75) == "INVESTIGAR"
```

Adicione em `tests/conftest.py` o fixture de sessão `trained(prepared)`, que roda `run_train(prepared)` e devolve o mesmo `Settings`. As Tarefas 10 e 12 o reutilizam. Depois escreva o teste de integração `test_analyze_batch_synthetic(trained)`, que deve verificar:
- `01`: `drifted_features == []`;
- `06`: `set(manifest["drifted_features"]) <= set(summary["drifted_features"])` (recall igual a 1);
- os arquivos `06_data_drift.html` e `06_performance.html` existem;
- `abs(psi_proprio - psi_evidently) < 0.05` para `AMT_INCOME_TOTAL`;
- um lote `06` sem a coluna `TARGET` produz `performance is None`.

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_drift.py -v`.
- [ ] **Passo 3: Implementar `stats.py` e `detect.py`.**
  Se, na base sintética, o recall do lote `06` não chegar a 1 por causa de uma feature cuja mudança é pequena demais (por exemplo `CNT_*`), revise o mapeamento `drifted_features` da Tarefa 7, não o limiar.
- [ ] **Passo 4: Rodar e ver passar.**
- [ ] **Passo 5: Commit.** `feat(drift): add PSI/KS statistics, Evidently reports, performance metrics and recommendation`

---

### Tarefa 10: Pipeline mensal, `simulate-all`, índice de relatórios e CLI

**Arquivos:**
- Criar: `src/credit_monitor/pipeline.py`, `src/credit_monitor/observability/report_index.py`, `tests/test_pipeline.py`
- Modificar: `src/credit_monitor/cli.py` (comandos `monitor --batch`, `simulate-all [--interval]`, `reset-metrics`)

**Interfaces:**
- `pipeline.BatchResult` (dataclass): `batch_id: str`, `status: Literal["ok", "blocked"]`, `summary_path: Path | None`, `recommendation: str | None`.
- `pipeline.run_monitor(settings: Settings, batch_id: str) -> BatchResult`. Etapas, cada uma dentro de `stage()`:
  1. `ingest`: lê `data/batches/<id>/batch.parquet` e pseudonimiza. Se o arquivo não existir, `BatchNotFound`.
  2. `contract`: `validate_batch`. Se falhar: `ctx.block`, relatório de falhas, `quarantine` (cópia), métricas `credit_contract_passed=0` e `credit_contract_violations{rule}`, e encerra com `blocked`.
  3. `transform`: `build_features`.
  4. `predict`: `load_model` + `score`.
  5. `drift`: `analyze_batch` contra `test_scored.parquet`.
  6. `publish`: todas as métricas da §7.3 a partir do resumo, mais `credit_model_auc_reference`, `credit_model_ks_reference`, `credit_pipeline_peak_memory_bytes`, `credit_pipeline_last_success_timestamp` e `credit_monitor_recommendation` (0/1/2); `publisher.push()`; run do MLflow no experimento `credit-monitoring` com as tags da spec §7.6 e os artefatos (HTMLs, JSONs, manifest, falhas); `build_index`.
  - O modelo é carregado **antes** do `ingest`. `ModelNotTrained` é propagado.
- `pipeline.run_simulate_all(settings: Settings, interval_seconds: float = 0.0) -> list[BatchResult]`: percorre `SIMULATION_ORDER`, com `time.sleep(interval)` entre lotes.
- `pipeline.reset_metrics(settings: Settings) -> int`: `delete_from_gateway` para cada lote de `SIMULATION_ORDER` e para `latest`. Retorna o número de grupos apagados.
- `report_index.build_index(reports_dir: Path) -> Path`: lê `reports/drift/*_summary.json` e `reports/contracts/*_failures.json` e gera `reports/index.html` com uma tabela por lote (status, features com drift, drift share, AUC, calibração, recomendação, links).
- CLI:
  - `monitor` retorna 0 para `ok`, 1 para `blocked` e 2 para `BatchNotFound`/`ModelNotTrained`, sempre com mensagem em português e sem traceback.
  - `simulate-all` retorna 0, mesmo com lotes bloqueados.
  - `--interval` aceita segundos (`30` ou `30s`).

- [ ] **Passo 1: Escrever o teste que falha.** Usa o fixture de sessão `trained` (Tarefa 9), com a infraestrutura em `None`.

```python
# tests/test_pipeline.py
import json, re
from credit_monitor.cli import main
from credit_monitor.pipeline import run_monitor, run_simulate_all

def test_simulate_all_end_to_end(trained):
    results = {r.batch_id: r for r in run_simulate_all(trained)}
    assert results["03-dirty"].status == "blocked"
    assert all(results[b].status == "ok" for b in ["01", "02", "03", "04", "05", "06"])
    assert results["01"].recommendation == "OK"
    assert results["06"].recommendation in {"INVESTIGAR", "RETREINAR"}
    assert (trained.reports_dir / "index.html").exists()
    assert (trained.data_dir / "quarantine/03-dirty.parquet").exists()

def test_blocked_batch_can_rerun(trained):
    assert run_monitor(trained, "03-dirty").status == "blocked"
    assert run_monitor(trained, "03-dirty").status == "blocked"

def test_no_raw_ids_leak(trained):
    import pandas as pd
    ids = set(pd.read_parquet(trained.data_dir / "reference/production_pool.parquet")["SK_ID_CURR"]
              .astype("int64").astype(str))
    for path in trained.reports_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".html"}:
            found = set(re.findall(r"\b9\d{8}\b", path.read_text(errors="ignore"))) & ids
            assert not found, f"ID bruto em {path}"

def test_cli_errors_are_friendly(trained, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(trained.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(trained.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(trained.models_dir))
    assert main(["monitor", "--batch", "99"]) == 2
    assert "Traceback" not in capsys.readouterr().err
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "vazio"))
    assert main(["monitor", "--batch", "01"]) == 2
    assert "train" in capsys.readouterr().out
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_pipeline.py -v`.
- [ ] **Passo 3: Implementar `pipeline.py`, `report_index.py` e os comandos.**
- [ ] **Passo 4: Rodar a suíte inteira.** `PYTEST -q`. Esperado: tudo passando.
- [ ] **Passo 5: Commit.** `feat(pipeline): add monthly monitor, simulate-all, metrics reset and HTML report index`

---

### Tarefa 11: Infraestrutura de observabilidade (compose, Prometheus, alertas, Loki, Grafana, MLflow)

**Arquivos:**
- Criar: `infra/mlflow/Dockerfile`, `infra/prometheus/prometheus.yml`, `infra/prometheus/alerts.yml`, `infra/loki/loki-config.yml`, `infra/grafana/provisioning/datasources/datasources.yml`, `infra/grafana/provisioning/dashboards/dashboards.yml`, `infra/grafana/dashboards/credit-monitoring.json`, `tests/test_infra_config.py`
- Modificar: `docker-compose.yml`, `.github/workflows/ci.yml` (job `promtool`), `Makefile` (`up`, `down`)

**Interfaces:**
- Consome `metrics.METRIC_CATALOG` (Tarefa 3) e os nomes de serviço e portas da spec §3.1.

**Serviços** (`docker compose up -d mlflow pushgateway prometheus grafana loki`):
- `mlflow`: imagem própria `python:3.12-slim` com `mlflow==2.17.2` e `prometheus-flask-exporter==0.23.1`. Comando: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts --expose-prometheus /mlflow/metrics`. Volume `mlflow-data`.
- `pushgateway`: `prom/pushgateway:v1.10.0`.
- `prometheus`: `prom/prometheus:v2.55.1`, com `--storage.tsdb.retention.time=395d` e `--web.enable-lifecycle`. Monta `infra/prometheus`.
- `loki`: `grafana/loki:3.2.1`, `-config.file=/etc/loki/loki-config.yml`, single-binary, filesystem, retenção de 2160 h (90 dias).
- `grafana`: `grafana/grafana:11.3.0`. Senha via `GF_SECURITY_ADMIN_PASSWORD`. Provisioning montado. Dashboard padrão `credit-monitoring`.
- `pipeline`: `depends_on` sem `condition`, para continuar rodando mesmo com a infraestrutura fora.
- Todas as portas no formato `"127.0.0.1:<porta>:<porta>"`.

**Prometheus** (`prometheus.yml`): `scrape_interval: 5s`, `rule_files: [alerts.yml]`. Jobs:
- `pushgateway` com `honor_labels: true`;
- `prometheus`;
- `mlflow` (`mlflow:5000`);
- `grafana` (`grafana:3000`);
- `loki` (`loki:3100`).

**Alertas** (`alerts.yml`): grupo `credit-monitoring` com os 9 alertas da spec §7.4.
- As expressões usam o seletor `{batch="latest"}`.
- `annotations.summary` em português, com `{{ $labels.feature }}`/`{{ $labels.rule }}` onde houver.
- `labels.severity`.

**Dashboard** (`credit-monitoring.json`, uid `credit-monitoring`): um painel por linha da spec §7.5. Datasources fixos por uid (`prometheus`, `loki`). Consultas obrigatórias:

| Painel | Tipo | Consulta |
|---|---|---|
| Último lote | stat | `credit_batch_info{batch="latest"}` (legenda `{{batch_id}}`) |
| Recomendação | stat com value mappings 0=OK verde, 1=INVESTIGAR amarelo, 2=RETREINAR vermelho | `credit_monitor_recommendation{batch="latest"}` |
| Alertas disparando | table | `ALERTS{alertstate="firing"}` |
| Status do contrato por lote | bar chart (instant, formato table) | `credit_contract_passed{batch!="latest"}` |
| Violações por regra | bar chart | `sum by (rule) (credit_contract_violations{batch!="latest"})` |
| PSI feature × lote | table + transformação *Grouping to matrix* (linhas = feature, colunas = batch) com cores 0,1/0,2 | `credit_feature_psi{batch!="latest"}` |
| Drift share e PSI do score | bar chart | `credit_dataset_drift_share{batch!="latest"}`, `credit_prediction_psi{batch!="latest"}` |
| AUC/KS vs referência | bar chart + threshold | `credit_model_auc{batch!="latest"}`, `credit_model_auc_reference{batch="latest"}` |
| Calibração | bar chart | `credit_default_rate_observed`, `credit_mean_predicted_pd` (`batch!="latest"`) |
| Viés | bar chart com threshold 0,8 | `credit_fairness_disparate_impact{batch!="latest"}` |
| Duração por etapa | bar chart | `credit_pipeline_stage_duration_seconds{batch="latest"}` |
| Linhas processadas | time series | `credit_pipeline_rows_processed` |
| Alvos no ar | state timeline | `up` |
| MLflow: requisições/latência | time series | as métricas `flask_http_request_*` expostas pelo MLflow |
| Memória do pipeline | bar chart | `credit_pipeline_peak_memory_bytes{batch!="latest"}` |
| Logs | logs (Loki) | `{job="credit_pipeline"} \| json \| level=~"$level"`, com variáveis `$level` e `$stage` |

- [ ] **Passo 1: Escrever o teste que falha** (consistência entre o código e a configuração)

```python
# tests/test_infra_config.py
import json, re, yaml
from pathlib import Path
from credit_monitor.observability.metrics import METRIC_CATALOG

INFRA = Path(__file__).resolve().parents[1] / "infra"

def _metric_names(text):
    return set(re.findall(r"\bcredit_[a-z_]+", text))

def test_alerts_reference_only_cataloged_metrics():
    rules = yaml.safe_load((INFRA / "prometheus/alerts.yml").read_text())
    names = {r["alert"] for g in rules["groups"] for r in g["rules"]}
    assert names == {"DataContractViolation", "PipelineStageFailed", "FeatureDriftDetected",
                     "DatasetDriftDetected", "PredictionDrift", "ModelPerformanceDegraded",
                     "ModelMiscalibrated", "FairnessDegraded", "InfraTargetDown"}
    exprs = " ".join(r["expr"] for g in rules["groups"] for r in g["rules"])
    assert _metric_names(exprs) <= set(METRIC_CATALOG)

def test_dashboard_references_only_cataloged_metrics():
    text = (INFRA / "grafana/dashboards/credit-monitoring.json").read_text()
    json.loads(text)
    assert _metric_names(text) <= set(METRIC_CATALOG)
```

Adicionar `pyyaml` às dependências `dev`.

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_infra_config.py -v`.
- [ ] **Passo 3: Criar os arquivos de infraestrutura e atualizar o compose.**
- [ ] **Passo 4: Rodar e ver passar.** Mesmo comando.
- [ ] **Passo 5: Validar as regras.**
  `docker run --rm -v ${PWD}/infra/prometheus:/p --entrypoint promtool prom/prometheus:v2.55.1 check rules /p/alerts.yml`. Esperado: `SUCCESS: 9 rules found`.
  Acrescentar esse mesmo comando ao CI, como job `promtool`.
- [ ] **Passo 6: Verificação de ponta a ponta com a base sintética**
  1. `docker compose up -d --build`
  2. `CLI prepare --synthetic`
  3. `CLI train`
  4. `CLI simulate-all`

  Esperado:
  - `http://127.0.0.1:9090/targets` com os 5 alvos `UP`;
  - `http://127.0.0.1:9090/alerts` com `FeatureDriftDetected`/`DatasetDriftDetected` disparando para `06`;
  - Grafana (`http://127.0.0.1:3000`) abre o dashboard com todos os painéis populados, inclusive os logs;
  - MLflow (`http://127.0.0.1:5000`) com os experimentos `credit-scoring-training` (modelo `credit-scoring@production`) e `credit-monitoring` (7 runs, um `blocked`).

  Rodar também `CLI monitor --batch 03-dirty` e conferir `DataContractViolation` disparando. Em seguida, `CLI monitor --batch 03`: o alerta resolve.
- [ ] **Passo 7: Commit.** `feat(observability): provision Prometheus alerts, Loki, Grafana dashboard and MLflow server`

---

### Tarefa 12: Análise causal

**Arquivos:**
- Criar: `src/credit_monitor/drift/causal.py`, `tests/test_causal.py`
- Modificar: `src/credit_monitor/cli.py` (comando `causal [--batch 06]`)

**Interfaces:**
- `causal.quantile_map(current: pd.Series, reference: pd.Series) -> pd.Series`
  - Numérica: posto percentual de cada valor não nulo no lote atual, mapeado para o quantil correspondente da referência. NaN fica NaN.
  - Categórica: reamostragem com as frequências da referência, com o mesmo `rng` fixo.
- `causal.decompose(pipeline, cur_df: pd.DataFrame, y_original: np.ndarray, y_new: np.ndarray, reference_auc: float) -> dict`: `auc_reference`, `auc_covariate_only`, `auc_total`, `drop_covariate = ref − covariate_only`, `drop_concept = covariate_only − total`.
- `causal.ablation(pipeline, ref_X: pd.DataFrame, cur_X: pd.DataFrame, y: np.ndarray, ref_scores: np.ndarray, cutoff: float, features: list[str]) -> pd.DataFrame`
  - Opera sobre a `model_matrix`.
  - Por feature, colunas: `feature, prediction_psi_after, auc_after, calibration_after, psi_recovery, auc_recovery`.
  - `psi_recovery = (psi_base − psi_after) / psi_base`. `auc_recovery` é análogo, sobre a queda em relação à referência.
- `causal.shap_importance(pipeline, X: pd.DataFrame) -> pd.Series`
  - `booster.predict(xgb.DMatrix(transformed), pred_contribs=True)`, média dos valores absolutos e descarte da coluna de bias.
  - Agrega as colunas one-hot de volta à feature original (pelos nomes de saída do `ColumnTransformer`).
- `causal.run_causal(settings: Settings, batch_id: str = "06") -> Path`
  - Lê o lote, `labels_original.parquet`, `test_scored.parquet` e o modelo.
  - Gera `reports/causal/causal_report.json` e `.html`, com a tabela da decomposição, a ablação ordenada por `psi_recovery` e o SHAP de referência vs lote.
  - Registra no MLflow (experimento `credit-monitoring`, run `causal-<batch>`).

- [ ] **Passo 1: Escrever o teste que falha**

```python
# tests/test_causal.py
import numpy as np, pandas as pd, pytest
from scipy.stats import spearmanr
from credit_monitor.drift.causal import quantile_map
from credit_monitor.drift.stats import psi

def test_quantile_map_restores_distribution_and_rank():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 20_000)); cur = pd.Series(rng.normal(2, 3, 20_000))
    cur[::10] = np.nan
    mapped = quantile_map(cur, ref)
    assert psi(ref, mapped.dropna()) < 0.02
    assert spearmanr(cur.dropna(), mapped.dropna()).correlation > 0.999
    assert mapped.isna().sum() == cur.isna().sum()

def test_run_causal_synthetic(trained):   # fixture de sessão da Tarefa 9
    import json
    from credit_monitor.drift.causal import run_causal
    path = run_causal(trained, "06")
    rep = json.loads(path.with_suffix(".json").read_text())
    d = rep["decomposition"]
    assert d["drop_concept"] > 0
    assert d["auc_total"] < d["auc_reference"]
    assert rep["ablation"][0]["psi_recovery"] >= rep["ablation"][-1]["psi_recovery"]
    assert path.with_suffix(".html").exists()
```

- [ ] **Passo 2: Rodar e ver falhar.** `PYTEST tests/test_causal.py -v`.
- [ ] **Passo 3: Implementar `causal.py` e o comando.**
- [ ] **Passo 4: Rodar e ver passar.**
- [ ] **Passo 5: Commit.** `feat(drift): add causal attribution (drift decomposition, quantile ablation, SHAP shift)`

---

### Tarefa 13: Execução com o dataset real, calibração e evidências

Pré-requisito: `data/raw/application_train.csv` presente, ou `KAGGLE_USERNAME`/`KAGGLE_KEY` no `.env`. Se faltar, pare e peça ao autor.

**Arquivos:**
- Modificar (só se a calibração exigir): `src/credit_monitor/config.py` (`DRIFT_SCHEDULE`)
- Criar: `docs/evidence/` (cópia dos relatórios e prints), `docs/evidence/RESULTADOS.md`

- [ ] **Passo 1: Rodar o fluxo real.**
  `docker compose up -d --build`, depois `CLI reset-metrics`, `CLI prepare`, `CLI train`, `CLI simulate-all`, `CLI causal`.
- [ ] **Passo 2: Checar os critérios de aceite.** Cada número vem de `reports/drift/*_summary.json` e `models/metadata.json`:

| Verificação | Esperado |
|---|---|
| Linhas removidas na limpeza da referência | > 0, com `income_range`, `domain_CODE_GENDER` e `annuity_not_null` presentes |
| AUC de referência | entre 0,72 e 0,78 (a spec espera 0,74–0,76) |
| Lote `01` | `drifted_features == []`, `OK` |
| Lote `03-dirty` | `blocked`, com as 6 regras esperadas |
| Lotes `05` e `06` | `RETREINAR`; queda de AUC > 0,05 no `06` |
| `06` | `calibration_ratio > 1.3` e recall 1 sobre o manifest |
| Causal | `drop_concept > 0` e as 3 principais features da ablação coerentes com o DAG |

- [ ] **Passo 3: Calibrar, se necessário.**
  Ajuste **só** o `DRIFT_SCHEDULE`: `concept_flip` se a queda de AUC for insuficiente; os fatores de inflação ou `young_weight` se o recall falhar. Depois rode `PYTEST -q` (os testes sintéticos não podem quebrar) e repita os Passos 1 e 2.
- [ ] **Passo 4: Gerar as evidências.**
  Copie `reports/index.html`, `reports/drift/*`, `reports/contracts/*` e `reports/causal/*` para `docs/evidence/`. Capture prints do Grafana (dashboard completo, heatmap de PSI, alertas disparando), do MLflow (comparação de runs) e do Prometheus `/alerts`, salvando em `docs/evidence/img/`. Escreva `RESULTADOS.md` com a tabela de critérios preenchida com os números reais.
- [ ] **Passo 5: Commit.** `docs(evidence): add real-data drift, contract and causal reports with dashboard screenshots`. Se o schedule mudou, faça antes um commit separado: `fix(drift): calibrate drift schedule for real data`.

---

### Tarefa 14: README de governança e roteiro do vídeo

**Arquivos:**
- Criar: `README.md`, `docs/roteiro-video.md`

- [ ] **Passo 1: Escrever o `README.md`** com as seções da spec §9, nesta ordem:
  1. **Problema.**
  2. **Arquitetura:** diagrama ASCII dos 6 serviços e do fluxo da spec §3.2.
  3. **Como rodar:** comandos `docker compose` para PowerShell, os dois caminhos do dataset e `prepare --synthetic` como demo sem Kaggle.
  4. **Etapas 1 a 3 com evidências:** links para `docs/evidence/` e prints.
  5. **Catálogo de métricas:** uma linha por entrada do `METRIC_CATALOG`, com tipo, labels, significado, limiar e alerta.
  6. **Governança e LGPD:** conteúdo integral da spec §8.1 e §8.2, incluindo a tabela de retenção, a base legal com a justificativa contra o consentimento, o art. 20, o art. 18, o RIPD resumido e a ressalva jurídica.
  7. **Viés:** resultados reais de `metadata.json["fairness"]` e da série `fairness_disparate_impact`.
  8. **Análise causal:** DAG em Mermaid, tabela da decomposição, top 5 da ablação, SHAP antes/depois, prescrição e ressalva.
  9. **Lições aprendidas.**
  10. **Vídeo:** link `_a preencher após a gravação_`.

  Todos os números citados vêm de `docs/evidence/RESULTADOS.md`.
- [ ] **Passo 2: Escrever o `docs/roteiro-video.md`:** quatro blocos STAR cronometrados que somam no máximo 5:00.
  - S, 0:00–0:40: degradação silenciosa.
  - T, 0:40–1:10: os três requisitos.
  - A, 1:10–3:30:
    - contrato bloqueando o `03-dirty` ao vivo;
    - `simulate-all --interval 20s` com o Grafana aberto;
    - heatmap de PSI;
    - plano LGPD.
  - R, 3:30–5:00: relatório do Evidently do `06`, alertas disparando, decomposição causal e lições aprendidas.
  - Cada bloco traz o comando exato e o painel ou URL a mostrar.
- [ ] **Passo 3: Verificar os links.**
  Todo link relativo do README aponta para um arquivo existente:
  `python -c "import re,pathlib;t=pathlib.Path('README.md').read_text(encoding='utf-8');[print('QUEBRADO',l) for l in re.findall(r'\]\(((?!http)[^)#]+)',t) if not pathlib.Path(l).exists()]"`
  Esperado: nenhuma saída.
- [ ] **Passo 4: Commit.** `docs: add README with governance, LGPD plan, metrics catalog, causal analysis and video script`
