# Roteiro do vídeo (STAR, até 5:00)

Quatro blocos cronometrados que somam exatamente 5:00. Cada bloco traz a fala, o comando exato (PowerShell,
na raiz do repositório) e o painel ou URL a mostrar. Números de referência: dataset real do Home Credit
(`docs/evidence/RESULTADOS.md`).

| Bloco | Tempo | Duração | Tema |
|---|---|---|---|
| **S**ituação | 0:00 a 0:40 | 40 s | Degradação silenciosa |
| **T**arefa | 0:40 a 1:10 | 30 s | Os três requisitos (mais a governança) |
| **A**ção | 1:10 a 3:30 | 2:20 | Contrato ao vivo, simulação, heatmap de PSI, plano LGPD |
| **R**esultado | 3:30 a 5:00 | 1:30 | Relatório do Evidently, alertas, causal e lições |

## Antes de gravar (preparo, fora do vídeo)

```powershell
docker compose up -d --build mlflow pushgateway prometheus grafana loki
docker compose run --rm pipeline prepare          # ou: prepare --synthetic
docker compose run --rm pipeline train
docker compose run --rm pipeline reset-metrics    # painéis vazios no início da demo
```

Os relatórios finais (`causal`, drift) já estão em `docs/evidence/`; para usar a versão gerada na sua
máquina, rode antes um `simulate-all` completo e o `causal`.

- **Portas.** Padrões: Grafana `http://localhost:3000` e Prometheus `http://localhost:9090` (MLflow 5000).
  Se houver conflito, o `.env` sobrescreve com `GRAFANA_PORT` e `PROMETHEUS_PORT`; nesse caso, use as portas
  do seu `.env` em todas as URLs abaixo.
- Deixe abertas, em abas: Grafana (dashboard "Credit Model Monitoring"), Prometheus `/alerts`,
  `docs/evidence/index.html`, `docs/evidence/drift/06_data_drift.html` e
  `docs/evidence/causal/causal_report.html`. Uma segunda janela de terminal fica pronta para o
  `simulate-all`.
- **Ensaie o tempo do `simulate-all --interval 20s`.** A execução completa (sete lotes mais as pausas)
  passa de 2 minutos; na edição, corte ou acelere a espera entre os lotes para que o bloco A caiba nos
  2:20.
- Dados já preparados e modelo já treinado: nada de `prepare` ou `train` durante a gravação.

---

## S: Situação (0:00 a 0:40)

**Mostrar:** slide com o título ou o dashboard do Grafana vazio, sem lotes.

**Fala (resumo):**

- "Um modelo de crédito é treinado uma vez e vai para produção. Depois a inflação sobe, o desemprego sobe e
  entra um perfil novo de cliente."
- "O modelo continua respondendo, sem erro e sem alarme de infraestrutura. Mas decide pior, e o rótulo, a
  inadimplência, só chega meses depois."
- "É a degradação silenciosa. Neste projeto eu monitoro um modelo de Credit Scoring do Home Credit, com
  307 mil propostas, e mostro como enxergá-la a tempo."

**Comando:** nenhum.

---

## T: Tarefa (0:40 a 1:10)

**Mostrar:** diagrama da arquitetura do README (seção 2) ou a lista abaixo na tela.

**Fala (resumo):** "A tarefa tem três requisitos técnicos, mais governança:"

1. "**Validar os dados** com um contrato que bloqueia lotes ruins antes de qualquer transformação (Pandera)."
2. "**Detectar drift** em seis lotes mensais simulados, de dados e de conceito, com Evidently e PSI/KS."
3. "**Observar** o pipeline, a infraestrutura e o modelo: Grafana, Prometheus, Loki e MLflow, com alertas."
4. "E documentar a **governança**: LGPD, viés e causalidade."

**Comando:** nenhum.

---

## A: Ação (1:10 a 3:30)

### A1. Contrato bloqueando o `03-dirty` ao vivo (1:10 a 1:50)

**Comando:**

```powershell
docker compose run --rm pipeline monitor --batch 03-dirty
```

**Mostrar:**

- A saída do terminal: "Lote 03-dirty bloqueado pelo contrato de dados" (código de saída 1).
- `docs/evidence/contracts/03-dirty_failures.html` (ou `reports/contracts/03-dirty_failures.html`): as
  **6 regras** violadas, `client_key_unique` (10), `ext_source_range`, `age_range`, `credit_positive`,
  `domain_NAME_CONTRACT_TYPE` e `income_not_null` (5 cada), sempre com `client_key`, nunca o ID bruto.
- Prometheus `http://localhost:9090/alerts`: `DataContractViolation` disparando.

**Fala:** "O lote chega sujo: renda nula, idade de 15 anos, IDs duplicados. O contrato bloqueia o lote
inteiro, manda para a quarentena, e o modelo nem chega a ser executado. O alerta dispara em segundos."

### A2. `simulate-all` com o Grafana aberto (1:50 a 2:50)

**Comando (segundo terminal):**

```powershell
docker compose run --rm pipeline simulate-all --interval 20s
```

**Mostrar:** Grafana `http://localhost:3000`, dashboard "Credit Model Monitoring", atualizando a cada lote:

- **Visão executiva:** "Último lote", "Recomendação do monitor" (de `OK` para `RETREINAR`) e "Alertas
  disparando".
- **Qualidade de dados:** o `03-dirty` aparece bloqueado, e o `03` seguinte passa.
- **Desempenho do modelo:** o AUC desce de 0,7585 (lote `01`) para 0,5833 (lote `06`), contra a linha da
  referência (0,7577).

**Fala:** "Seis meses simulados, com inflação, desemprego, perfil mais jovem e mudança no comportamento de
pagamento. O mês 3 chega sujo uma vez, e o simulador segue em frente. Cada lote publica métricas e logs."

### A3. Heatmap de PSI (2:50 a 3:10)

**Mostrar:** painel **"PSI feature × lote"** (a matriz PSI no Grafana) e o painel **"Drift share e PSI
do score"**.

**Fala:** "Cada linha é uma feature, cada coluna um lote. A renda, o preço do bem e o `EXT_SOURCE_2` esquentam
primeiro. O drift share passa de 0,30 no lote 05 e chega a 59% no 06. Uso o tamanho do efeito do KS, não o
p-valor, que satura com 20 mil linhas."

### A4. Plano LGPD (3:10 a 3:30)

**Mostrar:** README, seção 6 ("Governança e LGPD"): a tabela de privacy by design (cada medida com o arquivo
que a implementa), a base legal e a tabela de retenção.

**Fala:** "Privacidade por padrão: minimização com 21 de 122 colunas, `client_key` por HMAC, só agregados na
observabilidade, `CODE_GENDER` fora do modelo e retenção executável. A base legal é o art. 7º, X, proteção do
crédito, e explico por que não o consentimento. Há ainda o art. 20 com revisão humana e SHAP, o art. 18, o
RIPD e o fluxo de incidentes."

**Comando:** nenhum.

---

## R: Resultado (3:30 a 5:00)

### R1. Relatório do Evidently do lote 06 (3:30 a 3:55)

**Mostrar:** `docs/evidence/drift/06_data_drift.html` (relatório do Evidently) e a linha do `06` em
`docs/evidence/index.html`.

**Fala:** "No lote 06, 13 das 22 features estão com drift, 59,1%. A renda e o desemprego foram detectados. As
10 features que o manifest prometia foram todas detectadas, e nenhuma feature sem injeção foi sinalizada. O
AUC caiu de 0,7577 para 0,5833, e a calibração relativa é 2,16 vezes a da referência. Recomendação:
retreinar."

### R2. Alertas disparando (3:55 a 4:10)

**Mostrar:** Prometheus `http://localhost:9090/alerts` (ou o painel "Alertas disparando" do Grafana): 12
alertas, entre eles `FeatureDriftDetected` (8 features), `DatasetDriftDetected`, `PredictionDrift`,
`ModelPerformanceDegraded` e `ModelMiscalibrated`. `FairnessDegraded` fica inativo.

**Fala:** "Doze alertas disparam no estado final, sem nenhuma notificação manual."

### R3. Decomposição causal (4:10 a 4:35)

**Comando (opcional, se não estiver pronto; já foi gerado no preparo):**

```powershell
docker compose run --rm pipeline causal
```

**Mostrar:** `docs/evidence/causal/causal_report.html`: tabela de decomposição e ablação.

**Fala:** "Da queda de 0,1744 no AUC, 0,0242 vêm da mudança nas covariáveis e **0,1502, ou 86%, do concept
drift**. As `EXT_SOURCE` explicam a maior parte do deslocamento do score, mas nenhuma ablação recupera o AUC
nem a calibração. O estrago está nos rótulos. Isso só é causal porque a simulação controla o gerador dos
dados."

### R4. Lições aprendidas (4:35 a 5:00)

**Mostrar:** README, seção 9 (ou slide com os três pontos).

**Fala:**

1. "Medir o instrumento antes de culpar o dado: o PSI sobre valores redondos criava drift falso, e o KS é
   hipersensível em variável discretizada."
2. "Alerta precisa discriminar nível de degradação: o viés por idade, 0,54, já existe na referência, e o
   alerta acompanha a piora, enquanto o achado vai para a governança."
3. "Prescrição: retreino com dados recentes, features em termos reais e champion/challenger. Obrigado."

---

## Checklist final

- [ ] Duração total de 5:00 ou menos (confira no editor).
- [ ] Nenhum segredo à mostra: tampe o `.env` e a senha do Grafana.
- [ ] Link do vídeo colado na seção 10 do README.
