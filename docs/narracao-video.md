# Narração do vídeo (texto para gravar)

Texto falado, sincronizado com os cortes do vídeo já editado (4:50, sem áudio). Cada bloco começa no
instante em que a cena aparece na tela. O tempo de cada fala foi conferido com uma voz sintética a ritmo
normal (cerca de 175 palavras por minuto): todas cabem na janela, com folga de 0 a 5 s.

Dicas para gravar:

- Grave **um arquivo por bloco** (`01.m4a`, `02.m4a`...) ou uma tomada única; os dois servem para a montagem.
- Em blocos de ~6 s (lotes 01 a 06), fale sem pausa: a tela troca a cada 6,7 s.
- Números: fale "zero setenta e seis", não "zero vírgula setecentos e cinquenta e oito".
- Ambiente silencioso, celular a um palmo da boca, sem música.

| # | Início | Fim | Na tela | Fala |
|---|---|---|---|---|
| 01 | 0:00 | 0:15 | Slide de título | Olá! Este é o Tech Challenge da Fase 4: degradação silenciosa em credit scoring, num modelo de crédito treinado com as 307 mil propostas do Home Credit. |
| 02 | 0:15 | 0:35 | Grafana vazio | Um modelo de crédito é treinado uma vez e vai para produção. Aí a inflação sobe, o desemprego sobe, entra um cliente mais jovem. O modelo continua respondendo, sem erro e sem alarme de infraestrutura, mas decide pior. E a inadimplência só chega meses depois. |
| 03 | 0:35 | 1:05 | Slide "Tarefa" | A tarefa tem três frentes técnicas e uma de governança. Primeiro, validar os dados com um contrato Pandera, que bloqueia lotes ruins antes de qualquer transformação. Segundo, detectar drift de dados e de conceito em seis lotes mensais, com Evidently, PSI e KS. Terceiro, observar tudo com Grafana, Prometheus, Loki e MLflow. E, por fim, governança: LGPD, viés e análise causal. |
| 04 | 1:05 | 1:18 | Terminal `monitor --batch 03-dirty` | Começo pelo contrato. O lote três sujo chega com renda nula, idade de quinze anos e IDs duplicados. O contrato bloqueia o lote e manda para a quarentena; o modelo nem roda. |
| 05 | 1:18 | 1:31 | Relatório do contrato | O relatório do contrato mostra as seis regras violadas e uma amostra das falhas. Cada linha traz o client key, um pseudônimo por HMAC, nunca o ID bruto do cliente. |
| 06 | 1:31 | 1:45 | Prometheus, depois Grafana | No Prometheus, o alerta de violação de contrato dispara em segundos. E no Grafana, o painel de qualidade de dados já registra o lote bloqueado e as violações por regra. |
| 07 | 1:45 | 1:53 | Terminal `simulate-all` | Agora, a simulação completa: seis meses de inflação, desemprego e perfil novo de cliente. |
| 08 | 1:53 | 2:00 | Grafana, lote 01 | Lote um: AUC de zero setenta e seis, igual à referência. OK. |
| 09 | 2:00 | 2:06 | Lote 02 | No lote dois, renda e preço do bem começam a mudar, mas segue OK. |
| 10 | 2:06 | 2:13 | Lote 03-dirty | O mês três chega sujo, é bloqueado, e o simulador segue em frente. |
| 11 | 2:13 | 2:20 | Lote 03 | O três limpo passa, com AUC de zero setenta e dois. |
| 12 | 2:20 | 2:26 | Lote 04 | No quatro entra o concept drift: AUC de zero sessenta e quatro. Retreinar. |
| 13 | 2:26 | 2:33 | Lote 05 | No cinco, zero sessenta, com metade das features em drift. |
| 14 | 2:33 | 2:40 | Lote 06 | E no seis, AUC de zero cinquenta e oito, aprovando só 43 por cento. |
| 15 | 2:40 | 2:45 | Resumo no terminal | Resumo: três lotes OK, um bloqueado e três para retreinar. |
| 16 | 2:45 | 2:57 | Heatmap de PSI | No heatmap de PSI, cada linha é uma feature e cada coluna, um lote. O vermelho se espalha: renda e preço do bem cedo, os scores externos a partir do quatro. |
| 17 | 2:57 | 3:05 | Drift share | O drift share passa do limite de 30 por cento no lote cinco e chega a 59 no seis. |
| 18 | 3:05 | 3:25 | README §6 (LGPD) | Na governança, privacidade por padrão: só 21 das 122 colunas, client key por HMAC, só agregados na observabilidade, gênero fora do modelo e retenção automática. Base legal: proteção do crédito, com revisão humana pelo artigo vinte. |
| 19 | 3:25 | 3:42 | Evidently, lote 06 | No Evidently do lote seis, só pelo PSI, sete de 23 colunas têm drift. Somando o KS, o monitor sinaliza 13 de 22 features, e pega todas as dez que a simulação injetou, inclusive renda e desemprego. |
| 20 | 3:42 | 3:50 | Índice de relatórios | O índice de relatórios resume a trajetória: do OK ao retreinar, com o lote sujo bloqueado. |
| 21 | 3:50 | 3:58 | Alertas no Prometheus | No estado final, doze alertas disparam: drift, predição, desempenho e calibração. |
| 22 | 3:58 | 4:05 | Grafana, visão executiva | A visão executiva fecha em zero cinquenta e oito, contra setenta e seis. Retreinar. |
| 23 | 4:05 | 4:30 | Relatório causal | Mas por que caiu? A análise causal decompõe a queda de zero dezessete no AUC: só zero zero dois vêm da mudança nos dados. Zero quinze, ou 86 por cento, vêm do concept drift. Nenhuma ablação de feature recupera o AUC: o problema está nos rótulos. |
| 24 | 4:30 | 4:50 | Slide "Lições" | Três lições. Medir o instrumento antes de culpar o dado: PSI em valores redondos gerava drift falso. O alerta deve acompanhar a piora, não o viés que já existe na referência. E a prescrição: retreino com dados recentes e champion challenger. Obrigado! |

Fonte dos números: `docs/evidence/RESULTADOS.md` (AUC 0,7577 na referência; 0,7585, 0,7413, 0,7222,
0,6381, 0,5998 e 0,5833 nos lotes; drift share 50,0% e 59,1% nos lotes 05 e 06; queda total 0,1744, sendo
0,0242 de covariáveis e 0,1502 de concept drift).
