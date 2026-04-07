# Maintenance Priority Control Tower

## PT-BR

### Visão rápida
Este projeto mostra como transformar um modelo de manutenção preditiva em uma **control tower de priorização operacional**. Em vez de parar no score técnico de risco, ele combina probabilidade de falha com impacto de produção, criticidade de segurança e redundância operacional para gerar uma fila prática de manutenção.

### Problema de negócio
Em ambientes industriais, o ativo com maior risco técnico nem sempre é o primeiro que deve receber intervenção. A decisão real costuma depender de:
- impacto na produção;
- criticidade de segurança;
- redundância disponível;
- janela operacional;
- risco previsto de falha.

Este repositório modela esse problema ao criar um ranking final com:
- `risk_score`;
- `priority_score`;
- `priority_band`;
- `recommended_action`.

### Base pública de referência
O framing técnico do projeto usa como referência o **AI4I 2020 Predictive Maintenance Dataset**, da UCI. Para manter a execução local leve e reproduzível, o runtime usa uma amostra sintética `AI4I-style` expandida com sinais de criticidade operacional e impacto de negócio.

Referência oficial:
- [UCI - AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### O que o projeto faz
1. Gera uma base local de telemetria e criticidade por ativo.
2. Treina um classificador para prever `maintenance_required`.
3. Pontua a última janela de cada ativo.
4. Combina risco técnico e impacto operacional em `priority_score`.
5. Exporta uma tabela final de control tower para decisão.

### Estrutura do projeto
- `main.py`: entry point local.
- `src/sample_data.py`: gera a base sintética com sinais técnicos e operacionais.
- `src/modeling.py`: treina o pipeline e monta a tabela final de prioridade.
- `tests/test_project.py`: valida o contrato mínimo do projeto.
- `data/raw/public_dataset_reference.json`: referência pública usada no framing.
- `data/processed/maintenance_scored_cycles.csv`: holdout pontuado.
- `data/processed/maintenance_priority_tower.csv`: tabela final da control tower.
- `data/processed/maintenance_priority_control_tower_report.json`: relatório consolidado.

### Dados usados
- `asset_id`: identificador do ativo.
- `asset_type`: tipo do equipamento.
- `cycle`: posição temporal observada.
- `temperature`, `vibration`, `pressure`, `current`, `efficiency`, `throughput`, `noise_index`: sinais técnicos.
- `production_impact`: impacto de parada na produção.
- `redundancy_factor`: disponibilidade de backup ou redundância operacional.
- `safety_criticality`: nível de impacto em segurança.
- `maintenance_required`: alvo supervisionado.

### Modelagem
O pipeline usa:
- imputação de faltantes;
- `OneHotEncoder` para ativo e tipo;
- `RandomForestClassifier` com balanceamento por subsample.

Depois da etapa de predição, o projeto cria uma segunda camada de decisão:
- `risk_score`: risco técnico previsto;
- `priority_score`: combinação entre risco e impacto operacional;
- `priority_band`: `P1`, `P2`, `P3` ou `P4`;
- `recommended_action`: ação sugerida para a equipe.

### Resultados atuais
- `dataset_source = maintenance_control_tower_ai4i_style`
- `row_count = 807`
- `asset_count = 10`
- `positive_rate = 0.2912`
- `roc_auc = 0.9269`
- `average_precision = 0.9085`
- `f1 = 0.8430`
- `p1_assets = 6`

### Como executar
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### O que torna este projeto diferente
Este projeto vai além da pergunta “qual ativo tem maior chance de falhar?”.

Ele responde:
- qual ativo deve entrar primeiro na fila;
- qual ativo pode esperar;
- onde risco e impacto se combinam de forma crítica;
- qual ação operacional faz mais sentido.

### Do básico ao avançado
No nível básico, o projeto é um classificador de manutenção.

No nível intermediário, ele é um sistema de ranking de ativos.

No nível avançado, ele permite discutir:
- priorização operacional;
- control towers industriais;
- separação entre risco técnico e decisão de negócio;
- governança de score;
- monitoramento de backlog e criticidade.

### Batch vs stream
- `batch`:
  - recalcula o ranking completo da frota;
  - atualiza backlog de manutenção;
  - gera relatórios de turno e campanha.

- `stream`:
  - reage à nova telemetria;
  - promove um ativo para `P1` em baixa latência;
  - atualiza painéis operacionais em tempo quase real.

### Governança e monitoramento
Uma control tower real precisa monitorar:
- drift de telemetria;
- variação de criticidade por ativo;
- backlog de `P1` e `P2`;
- tempo médio até intervenção;
- divergência entre prioridade prevista e intervenção executada.

### Limitações
- a execução local usa uma amostra sintética inspirada em manutenção industrial;
- a camada de criticidade operacional é simulada;
- o fluxo representa apoio à decisão, não despacho real de manutenção.

## EN

### Quick overview
This project turns predictive maintenance into an operational **maintenance priority control tower**. Instead of stopping at technical failure risk, it combines predicted risk with production impact, safety criticality, and redundancy to generate a practical maintenance queue.

### Public dataset framing
The project is technically framed around the **AI4I 2020 Predictive Maintenance Dataset** from UCI. Runtime execution uses a compact local `AI4I-style` sample expanded with operational-priority signals.

### What the project does
1. Generates local telemetry and criticality data.
2. Trains a classifier for `maintenance_required`.
3. Scores the latest window of each asset.
4. Builds a `priority_score` using technical and operational signals.
5. Exports a control tower table for action.

### Current results
- `dataset_source = maintenance_control_tower_ai4i_style`
- `row_count = 807`
- `asset_count = 10`
- `positive_rate = 0.2912`
- `roc_auc = 0.9269`
- `average_precision = 0.9085`
- `f1 = 0.8430`
- `p1_assets = 6`

### Run locally
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### Advanced discussion points
This repository is useful to discuss:
- predictive maintenance plus decision support;
- technical risk versus operational priority;
- batch versus near-real-time prioritization;
- control tower governance and monitoring.
