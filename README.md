# Exploração de preços da cesta básica (set/2022 a ago/2026)

Trabalho de extensão: exploração dos preços de itens da cesta básica coletados mensalmente em
cerca de 28 supermercados.

## Como executar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./rodar_tudo.sh          # ~4 min (a Q1 é a etapa mais lenta)
```

## Estrutura

```
dados/bruto/ICB_DADOS_2026.xlsx     base original (não alterada)
dados/processado/                   gerado pelo 01_preparacao.py
src/
  utils.py                          caminhos, paleta e estilo dos gráficos
  01_preparacao.py                  limpeza, harmonização e índices de preço
  02_exploratoria.py                análise exploratória (EDA)
  03_q1_previsao.py                 Questão 1 - previsão de preços
  04_q2_mercados.py                 Questão 2 - liderança e convergência entre supermercados
  05_q3_produtos.py                 Questão 3 - um produto antecipa outro?
resultados/figuras/                 gráficos (PNG)
resultados/tabelas/                 tabelas (CSV), prefixadas por etapa (prep_, eda_, q1_, q2_, q3_)
RESULTADOS.md                       resumo das questões, métodos e conclusões
```

Os resultados são determinísticos (sementes fixas), então rodar o pipeline de novo produz os
mesmos números.
