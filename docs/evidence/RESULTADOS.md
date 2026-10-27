# Resultados da execução com o dataset real (Home Credit)

Execução completa do fluxo sobre o `application_train.csv` real do Kaggle (307.511 linhas), com a
infraestrutura no ar (MLflow, Pushgateway, Prometheus, Grafana, Loki). Todos os números abaixo vêm de
`drift/*_summary.json`, `causal/causal_report.json`, `contracts/*.json` e `models/metadata.json`
(cópias neste diretório, exceto os metadados do modelo, que ficam em `models/`).

Esta é a segunda execução, refeita depois de duas correções (seção 6): bins de PSI robustos a empates e
alerta de viés relativo à referência. Fluxo executado (PowerShell, na raiz do repositório):

```powershell
docker compose up -d --build mlflow pushgateway prometheus grafana loki
docker compose run --rm pipeline reset-metrics
docker compose run --rm pipeline prepare
docker compose run --rm pipeline train
docker compose run --rm pipeline simulate-all
docker compose run --rm pipeline causal
```

Os rótulos dos lotes `04`–`06` são sintéticos (concept drift injetado pela simulação); a base original não
foi alterada.

## 1. Critérios de aceite

| Verificação | Esperado | Obtido | Resultado |
|---|---|---|---|
| Linhas removidas na limpeza da referência | > 0, com `income_range`, `domain_CODE_GENDER` e `annuity_not_null` | 19 de 307.511 removidas: `annuity_not_null` 12, `domain_CODE_GENDER` 4 (valor `XNA`), `income_range` 3 | Atendido |
| AUC de referência | entre 0,72 e 0,78 (spec: 0,74–0,76) | 0,7577 (teste); KS 0,390; PR-AUC 0,239; Brier 0,186; corte 0,4494 | Atendido |
| Lote `01` | `drifted_features == []` e `OK` | `drifted_features == []`, drift share 0%, PSI do score 0,0004, recomendação `OK` (o impacto disparato de idade, 0,515, não piorou em relação à referência, 0,541) | Atendido |
| Lote `03-dirty` | `blocked`, com as 6 regras esperadas | Bloqueado: `client_key_unique` (10), `ext_source_range` (5), `age_range` (5), `credit_positive` (5), `domain_NAME_CONTRACT_TYPE` (5), `income_not_null` (5) | Atendido |
| Lotes `05` e `06` | `RETREINAR`; queda de AUC > 0,05 no `06` | `05` e `06` = `RETREINAR`; AUC do `06` = 0,5833, queda de 0,1744 em relação à referência (0,7577) | Atendido |
| `06`: calibração | `calibration_ratio` > 1,3 (relativa à referência) | 2,156 (bruta: 0,440; referência ≈ 0,204) | Atendido |
| `06`: recall sobre o manifest | 1 | 10 de 10 features de `drifted_features` detectadas (recall 1,00) | Atendido |
| `06`: renda e mais uma variável importante detectadas | `AMT_INCOME_TOTAL` e desemprego/outra | `AMT_INCOME_TOTAL` detectada (PSI 0,654; KS 0,148). Desemprego: `YEARS_EMPLOYED` (PSI 0,200; KS 0,168), `IS_UNEMPLOYED_OR_PENSIONER` (KS 0,116) e `NAME_INCOME_TYPE` (PSI 1,230) detectadas | Atendido |
| Causal | `drop_concept > 0`; 3 principais da ablação coerentes com o DAG | `drop_concept` = 0,1502; principais: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `YEARS_EMPLOYED` | Atendido |
| Testes | suíte verde | 176 passed (`pytest -q` no contêiner) | Atendido |

## 2. Resultado por lote

| Lote | Features com drift | Drift share | PSI do score | AUC | KS | Calibração (relativa) | Taxa de aprovação | Impacto disparato gênero / idade | Recomendação |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 0 | 0,0% | 0,000 | 0,7585 | 0,394 | 0,998 | 62,3% | 0,865 / 0,515 | OK |
| 02 | 2 | 9,1% | 0,008 | 0,7413 | 0,357 | 1,053 | 64,9% | 0,889 / 0,588 | OK |
| 03-dirty | bloqueado pelo contrato (não pontuado) | | | | | | | | |
| 03 | 5 | 22,7% | 0,011 | 0,7222 | 0,328 | 0,987 | 64,2% | 0,890 / 0,592 | OK |
| 04 | 6 | 27,3% | 0,057 | 0,6381 | 0,201 | 1,476 | 55,8% | 0,845 / 0,576 | RETREINAR |
| 05 | 11 | 50,0% | 0,162 | 0,5998 | 0,144 | 1,847 | 48,8% | 0,861 / 0,546 | RETREINAR |
| 06 | 13 | 59,1% | 0,298 | 0,5833 | 0,124 | 2,156 | 43,1% | 0,827 / 0,530 | RETREINAR |

PSI por lote de três features (as duas primeiras concentradas em valores redondos):

| Feature | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|
| AMT_INCOME_TOTAL | 0,001 | 0,001 | 0,489 | 0,502 | 0,533 | 0,654 |
| AMT_GOODS_PRICE | 0,001 | 0,394 | 0,300 | 0,262 | 0,232 | 0,490 |
| AMT_ANNUITY | 0,001 | 0,019 | 0,059 | 0,135 | 0,208 | 0,274 |

Inadimplência observada x PD média prevista: `01` 8,1% x 39,6%; `04` 12,9% x 42,9%; `05` 17,4% x 46,2%;
`06` 21,5% x 48,8%. A razão bruta (observado/previsto) é 0,204 na referência porque o modelo foi
treinado com `scale_pos_weight` (PD inflada); por isso o critério usa a razão relativa à referência.

Lotes de 20.000 linhas cada. Concept drift aplicado (rótulos virados): `04` 899, `05` 1.723, `06` 2.518.

Features com drift em cada lote (regra: PSI ≥ 0,20 ou KS D > 0,10); as listas não mudaram em relação à primeira execução:

- `02`: AMT_GOODS_PRICE, AMT_INCOME_TOTAL
- `03`: AMT_ANNUITY, AMT_GOODS_PRICE, AMT_INCOME_TOTAL, CREDIT_TERM, NAME_INCOME_TYPE
- `04`: AMT_ANNUITY, AMT_GOODS_PRICE, AMT_INCOME_TOTAL, CREDIT_TERM, EXT_SOURCE_2, NAME_INCOME_TYPE
- `05`: AGE_YEARS, AMT_ANNUITY, AMT_GOODS_PRICE, AMT_INCOME_TOTAL, ANNUITY_INCOME_RATIO, CREDIT_TERM,
  EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, NAME_INCOME_TYPE, YEARS_EMPLOYED
- `06`: AGE_YEARS, AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE, AMT_INCOME_TOTAL, ANNUITY_INCOME_RATIO,
  CREDIT_TERM, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, IS_UNEMPLOYED_OR_PENSIONER, NAME_INCOME_TYPE,
  YEARS_EMPLOYED

Recall sobre `drifted_features` do manifest (conjunto esperado como detectável): `03` 1/1, `04` 2/2,
`05` **7/9** (não detectadas: `AMT_CREDIT` e `IS_UNEMPLOYED_OR_PENSIONER`), `06` 10/10. O critério de aceite
só exige recall 1 no `06`.

## 3. Lote 06: drift injetado x detectado

`injected_features` é a verdade-base (toda feature tocada pelos parâmetros); a coluna "esperada" marca
as que `drifted_features` do manifest promete como detectáveis.

| Feature injetada | PSI | KS (D) | Esperada | Detectada |
|---|---|---|---|---|
| AGE_YEARS | 0,233 | 0,221 | sim | sim |
| AMT_ANNUITY | 0,274 | 0,225 | sim | sim |
| AMT_CREDIT | 0,171 | 0,123 | sim | sim (KS) |
| AMT_GOODS_PRICE | 0,490 | 0,162 | sim | sim |
| AMT_INCOME_TOTAL | 0,654 | 0,148 | não | sim (PSI e KS) |
| ANNUITY_INCOME_RATIO | 0,104 | 0,137 | não | sim (KS) |
| CREDIT_INCOME_RATIO | 0,008 | 0,042 | não | **não** (infla junto com a renda; efeito líquido pequeno) |
| CREDIT_TERM | 0,382 | 0,251 | sim | sim |
| EXT_SOURCE_1 | 0,071 | 0,123 | não | sim (KS) |
| EXT_SOURCE_2 | 1,328 | 0,300 | sim | sim |
| EXT_SOURCE_3 | 0,672 | 0,231 | sim | sim |
| IS_UNEMPLOYED_OR_PENSIONER | 0,075 | 0,116 | sim | sim (KS) |
| NAME_INCOME_TYPE | 1,230 | n/a (categórica) | sim | sim |
| YEARS_EMPLOYED | 0,200 | 0,168 | sim | sim |

14 injetadas, 13 detectadas (todas as 10 esperadas, mais 3 não prometidas); nenhuma feature sem injeção foi
sinalizada (zero falso positivo sobre o conjunto não tocado).

## 4. Análise causal (lote 06)

Decomposição da queda de AUC (referência 0,7577):

| Etapa | AUC | Queda |
|---|---|---|
| AUC na referência | 0,7577 | |
| AUC(X₀₆, rótulos originais): efeito da mudança em X | 0,7335 | 0,0242 (mudança nas covariáveis) |
| AUC(X₀₆, rótulos novos): efeito total | 0,5833 | 0,1744 no total |
| Contribuição do concept drift (diferença entre os dois últimos) | | **0,1502** (86% da queda) |

Top 5 da ablação (feature volta à distribuição de referência por mapeamento de quantis; PSI do score
no `06` sem ablação = 0,298):

| # | Feature | PSI do score após | Recuperação do PSI | AUC após | Recuperação de AUC | Calibração após |
|---|---|---|---|---|---|---|
| 1 | EXT_SOURCE_2 | 0,081 | 73,0% | 0,5867 | +0,0193 | 2,374 |
| 2 | EXT_SOURCE_3 | 0,130 | 56,6% | 0,5838 | +0,0029 | 2,329 |
| 3 | YEARS_EMPLOYED | 0,245 | 17,9% | 0,5882 | +0,0280 | 2,199 |
| 4 | EXT_SOURCE_1 | 0,249 | 16,4% | 0,5844 | +0,0064 | 2,191 |
| 5 | AMT_CREDIT | 0,254 | 14,8% | 0,5822 | -0,0066 | 2,210 |

Leitura pelo DAG: as `EXT_SOURCE` (canal digital, perfil jovem com histórico curto) respondem pela maior
parte do deslocamento do score; `YEARS_EMPLOYED` vem do ramo desaceleração, desemprego. As features `AMT_*`
não são irrelevantes: o top 5 só ordena quem mais recupera o PSI do score quando restaurado sozinho
(`AMT_CREDIT` é o 5º, 14,8%, com recuperação de AUC negativa, -0,0066; `AMT_ANNUITY` vem logo atrás, 14,6%).
Recuperação de PSI **negativa** não é ruído; é consistente com deslocamentos que se compensam parcialmente:
restaurar sozinha `AMT_GOODS_PRICE` (-23,7%, PSI do score vai a 0,369) ou `AGE_YEARS` (-15,8%, vai a 0,346)
piora o PSI do score, e desfazer só um deslocamento desfaz parte do cancelamento (a ablação não identifica o mecanismo). A renda sozinha recupera 0,8%. Nenhuma ablação recupera a calibração nem o AUC de forma relevante
(calibração permanece em 2,2–2,4x): o estrago principal está no concept drift (rótulos), não nas
covariáveis, o que justifica o retreino com dados recentes. |SHAP| médio, referência vs `06`: queda de
`EXT_SOURCE_2` (0,377 para 0,267), `EXT_SOURCE_3` (0,382 para 0,327) e `EXT_SOURCE_1` (0,189 para 0,162);
o modelo perde as variáveis em que mais se apoia. As intervenções só têm leitura causal porque a
simulação controla o gerador dos dados.

## 5. Modelo e viés (referência, `models/metadata.json`)

| Atributo | Taxa de aprovação por grupo | Impacto disparato | Diferença de equal opportunity |
|---|---|---|---|
| Gênero | F 65,8%, M 56,7% | 0,861 (≥ 0,8) | 0,080 |
| Faixa etária | <30 42,8%; 30–45 59,3%; 45–60 69,8%; 60+ 79,1% | **0,541 (< 0,8)** | 0,340 |

TPR (aprovação entre bons pagadores): F 0,685 e M 0,605; <30 0,464, 30–45 0,629, 45–60 0,721, 60+ 0,803.
Teste de proxy (informação mútua com `CODE_GENDER`, top 5): FLAG_OWN_CAR 0,057, CREDIT_TERM 0,031,
AMT_INCOME_TOTAL 0,022, EXT_SOURCE_1 0,021, NAME_FAMILY_STATUS 0,019.

## 6. Correções aplicadas depois da primeira execução

A primeira execução com os dados reais expôs dois problemas, corrigidos no commit
`fix(drift): tie-robust PSI bins and fairness degradation relative to reference`.

**A. PSI com valores empatados.** `AMT_INCOME_TOTAL` e `AMT_GOODS_PRICE` concentram-se em valores redondos
(112.500, 135.000, 157.500...), e os cortes de quantil caíam exatamente sobre essas massas: até a inflação de
2% do lote `02` trocava de bin massas inteiras (PSI da renda 0,491). Agora cada borda numérica fica no meio
entre o corte e o próximo valor da grade de percentis da referência acima dele. Usar apenas o próximo valor
distinto não bastava (testado: a borda ficava a cerca de 0,4% do corte, por causa de valores raros avulsos, e
o PSI do +2% continuava em 0,47). Resultado sobre a renda real, referência contra referência inflada:

| Inflação uniforme | PSI antes | PSI depois |
|---|---|---|
| +2% | 0,470 | 0,001 |
| +5% | 0,496 | 0,473 |
| +12% | 0,193 | 0,620 |

PSI da renda por lote (inflação de 0, 2, 5, 8, 10 e 12%): antes 0,002; 0,491; 0,505; 0,571; 0,247; 0,189
(não monotônico); depois 0,001; 0,001; 0,489; 0,502; 0,533; 0,654 (monotônico). Teste unitário: série com
poucos níveis redondos mais valores avulsos; x1,02 dá PSI < 0,10 e x1,25 dá PSI ≥ 0,20 (o teste falha com a
borda "próximo valor distinto"). Limiares inalterados. O teste de consistência com o Evidently e os gates de
`DRIFT_EFFECTS` continuaram válidos na base sintética (176 testes passam).

Alcance da correção: ela remove só o artefato "massa de empate sobre a borda", e apenas para deslocamentos
menores que a distância entre dois níveis de preço consecutivos. Acima disso (por exemplo +5%), a massa cruza
a borda de verdade e o PSI sobe, o que é medição legítima. A série "monotônica" vale para a renda; o
`AMT_GOODS_PRICE`, também de valores redondos, segue não monotônico (seção 8, item 3).

Ressalva restante: o KS continua alto em variável empatada (renda no `02`: D = 0,119, com PSI 0,001), porque
deslocar uma massa de empate move a CDF de uma vez. Por isso a renda ainda é marcada como drift já no `02`,
agora só pelo KS, e o `FeatureDriftDetected` (PSI ≥ 0,20) do Prometheus só a lista a partir do `03`.

**B. Viés: alerta de piora, não do nível-base.** O impacto disparato de faixa etária já é 0,54 na referência
(seção 7), então a regra absoluta (< 0,8) mantinha `INVESTIGAR` e o `FairnessDegraded` ligados desde o lote
`01`. Agora a recomendação e o alerta só disparam se `DI < 0,8` **e** `DI < DI da referência - 0,05`
(`FAIRNESS_DEGRADATION_MARGIN`). O resumo do lote traz `fairness` (absoluto) e `fairness_reference`; o
Pushgateway recebe `credit_fairness_disparate_impact_reference{attribute}`; o painel de viés mostra a
referência em barras cinzas ao lado do valor do lote, com a linha de 0,8. Efeito: `01`, `02` e `03` voltam a
`OK`, e `FairnessDegraded` fica inativo no `06` (idade 0,530 contra 0,541 na referência: queda de 0,011,
menor que a margem de 0,05). A expressão do alerta foi validada com `promtool check rules` e contra o
Prometheus (com a margem invertida, ela retorna `age_band`).

## 7. Achado de governança permanente

O modelo de base tem impacto disparato por faixa etária de **0,54** no conjunto de teste da referência,
abaixo do piso de 0,8 da regra dos quatro quintos. Taxas de aprovação por faixa (`models/metadata.json`):

| Faixa etária | Taxa de aprovação | TPR (aprovação entre bons pagadores) |
|---|---|---|
| < 30 | 42,8% | 0,464 |
| 30–45 | 59,3% | 0,629 |
| 45–60 | 69,8% | 0,721 |
| 60+ | 79,1% | 0,803 |

Diferença de equal opportunity por idade: 0,340 (gênero: impacto disparato 0,861, diferença 0,080). Os jovens
inadimplem mais na base real, então o modelo aprova menos jovens, mas isso é um risco jurídico e ético que não
some com o monitoramento: é um achado do modelo-base, não do drift. Mitigações documentadas na spec (seção
8.3): ajuste de corte por grupo (com discussão jurídica sobre tratamento diferenciado por idade) ou
reponderação no retreino. O alerta mensal passa a acompanhar a degradação em relação a esse nível-base (piora
de mais de 0,05 e abaixo de 0,8); o próprio nível-base exige decisão de governança antes de ir a produção.

## 8. Observações e ressalvas

1. **Calibração do cronograma: nenhuma.** `DRIFT_SCHEDULE`, `DRIFT_EFFECTS` e limiares ficaram como estavam.
2. O recall do `05` é 7/9: os gates de `AMT_CREDIT` (`credit_infl` ≥ 0,20) e
   `IS_UNEMPLOYED_OR_PENSIONER` (`unemployed_target` ≥ 0,28) foram calibrados para a base sintética e são
   otimistas nos dados reais; no `06` ambas são detectadas (por KS).
3. `AMT_GOODS_PRICE` também tem PSI não monotônico (0,394 no `02`, depois 0,300, 0,262, 0,232 e 0,490 no
   `06`): com `credit_infl` de 4% já cruza bordas, e os preços do bem são valores redondos.
4. Alertas do Prometheus disparando no estado final (`latest` = `06`), 12 no total: `FeatureDriftDetected`
   (8: AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE, CREDIT_TERM, EXT_SOURCE_2, EXT_SOURCE_3,
   NAME_INCOME_TYPE, AGE_YEARS), `DatasetDriftDetected` (0,591), `PredictionDrift` (0,298),
   `ModelPerformanceDegraded` (AUC 0,583) e `ModelMiscalibrated` (2,156). `FairnessDegraded`,
   `DataContractViolation`, `PipelineStageFailed` e `InfraTargetDown` ficam inativos.
5. Recursos: o pipeline usa no máximo cerca de 470 MiB por lote (painel "Memória de pico"); a leitura das
   307 mil linhas do CSV no `prepare` terminou em cerca de 26 s, sem estouro de memória (Docker Desktop
   com 15,5 GiB).
6. Privacidade: nenhum identificador bruto da base nos arquivos deste diretório (varredura por pertinência
   ao conjunto de IDs): nenhum valor da coluna aparece como token numérico de 6 dígitos nos `.json`/`.md`;
   nos `.html` do Evidently só coincidem constantes do código JavaScript/CSS embutido (`131070`, `191970`),
   idênticas em todos os arquivos e sem relação com a base.
7. O MLflow mantém o histórico de execuções anteriores (inclusive sintéticas) no experimento
   `credit-monitoring`; a comparação das capturas usa só os seis runs mais recentes (`01`–`06`).

## 9. Capturas de tela (`img/`)

| Arquivo | Conteúdo |
|---|---|
| `img/grafana-dashboard-completo.jpg` | Dashboard completo do Grafana (último lote 06, `RETREINAR`; painel de viés com a referência em cinza) |
| `img/grafana-psi-matriz.jpg` | Matriz PSI feature x lote |
| `img/prometheus-alerts.jpg` | Página `/alerts` do Prometheus, 12 alertas disparando (`FairnessDegraded` inativo) |
| `img/mlflow-comparacao-runs.jpg` | Comparação dos runs `01`–`06` no MLflow (experimento `credit-monitoring`) |

Relatórios: `index.html` (tabela por lote, com links relativos para `drift/` e `contracts/`),
`drift/`, `contracts/` e `causal/`.
