# Resumo dos resultados

Base para o relatório. Todos os números vêm de `resultados/tabelas/`, e os gráficos citados
estão em `resultados/figuras/`.

---

## 0. Dados e preparação (`01_preparacao.py`)

**Base:** 46.264 registros (mês, data de coleta, estabelecimento, produto, marca, preço,
quantidade, preço por kg/L = PPK), set/2022 a ago/2026 (48 meses), cerca de 28 supermercados.

**Problemas encontrados e tratamento** (funil em `prep_funil.csv`):

| Problema | Tratamento | Linhas |
|---|---|---|
| 1.743 linhas **duplicadas exatas** (923 só em dez/2024) | removidas | 46.264 → 44.521 |
| Nomes trocados em mar/2026 ("Pão", "Farinha", "Carne Acem"…) | mapeados para o nome original | — |
| Cortes de carne com 1 a 2 meses (Alcatra, Patinho, Músculo) | removidos (sem série) | → 44.507 |
| Estabelecimentos `ALT_` (substitutos de um único mês) | removidos | → 44.392 |
| Erros de digitação (R$ 0,10), embalagem errada, margarina como "Manteiga" | PPK fora de [mediana/2, 2×mediana] do mês-produto | → 44.047 (−0,8%) |
| Marca registrada como o número `0` em dez/2024 | tratada como "sem marca" | — |

**Duas séries de preço por produto:**
1. **Mediana entre lojas** (R$/kg): o preço típico de prateleira. É o alvo da Q1.
2. **Índice encadeado "matched-model" (Jevons):** variação média do log-preço do *mesmo item*
   (loja + produto + marca + embalagem) entre meses consecutivos. Elimina o efeito de troca de
   marca/embalagem. É usado na Q2 e na Q3.
   - A *mediana* das variações não serve: de 14% (Tomate) a 63% (Pão Francês) dos itens não mudam
     de preço de um mês para o outro (`prep_rigidez_precos.csv`), e a mediana fica presa em 0.
   - O índice encadeado tem menos ruído (desvio-padrão das variações menor em todos os produtos,
     `eda_mediana_vs_encadeado.csv`), mas sofre *chain drift* em perecíveis (ex.: Batata). Por
     isso não é usado como nível.

**Exploratória (`02_exploratoria.py`):**
- Variação no período (`eda_variacao_acumulada.png`): Café +56%, Tomate +36%, Banana Prata +25%,
  Batata +23%, Pão +21%; Farinha de Trigo −15% e Óleo −7%.
- Dispersão entre lojas (CV): de 9% (Pão) a 26% (Batata). Os perecíveis são os mais dispersos.
- Testes ADF/KPSS (`eda_estacionariedade.csv`): log-preços em geral não estacionários; as
  variações mensais são estacionárias. Por isso as variações são modeladas.
- Sazonalidade fraca e inconsistente com só 4 anos (`eda_sazonalidade.png`).

---

## Questão 1: É possível prever o preço dos itens da cesta 1 e 3 meses à frente?

**Questão.** Dado o histórico de preços, conseguimos prever o preço mediano de cada produto em
t+1 e t+3 melhor que a regra "o preço do mês que vem é igual ao de hoje" (passeio aleatório)?
Usar os preços de *outros* produtos melhora a previsão?

**Objetivo e motivação.** Antecipar o custo da cesta é útil para orçamento familiar, programas
sociais e planejamento de compras de entidades (restaurantes populares, bancos de alimentos).
Saber *se* o preço é previsível também é um resultado: se não for, a melhor previsão disponível
é o preço atual.

**Método** (`03_q1_previsao.py`):
- Alvo: log da mediana entre lojas, 14 produtos com série completa.
- **Validação por origem móvel com janela expansível**: os últimos 18 meses (mar/2025 a
  ago/2026) são o teste. Em cada origem, os modelos são reajustados usando apenas dados até
  aquele mês (sem vazamento).
- Modelos:
  - de referência: ingênuo (passeio aleatório), ingênuo sazonal e drift;
  - estatísticos: SES, Holt amortecido e ARIMA(p,1,q) com p,q ≤ 2 escolhidos por AICc a cada
    origem;
  - aprendizado supervisionado global (um modelo para os 14 produtos): Ridge e LightGBM com
    lags de variação, desvio da média de 12 meses, volatilidade, variação no mesmo mês do ano
    anterior e mês do ano. Cada um tem duas versões, só com o próprio produto e com as
    variações dos outros 13 produtos. Hiperparâmetros (α do Ridge; folhas e árvores do
    LightGBM) escolhidos por validação temporal interna nos últimos 6 meses do treino;
  - combinação: média de Holt, ARIMA e Ridge.
- Métricas: MAPE, MASE e **AvgRelMAE** (média geométrica, entre produtos, do MAE relativo ao
  passeio aleatório; < 1 = melhor). **Teste de Diebold-Mariano** (com correção HLN) contra o
  passeio aleatório.

**Resultados** (`q1_metricas.csv`, `q1_avgrelmae.png`, `q1_relmae_produtos.png`):

| h | Melhor modelo | AvgRelMAE | DM p-valor |
|---|---|---|---|
| 1 | SES / Combinação | 0,996 | 0,95 |
| 3 | ARIMA (AICc) | 1,001 | 0,63 |

- **Nenhum modelo supera o passeio aleatório de forma significativa**, em nenhum horizonte.
- Ridge e LightGBM são **significativamente piores** em h=1 (DM p < 0,05). O ingênuo sazonal é
  o pior (AvgRelMAE 2,5).
- **Os outros produtos não ajudam**: Ridge 1,058 → 1,062 e LightGBM 1,112 → 1,092 (h=1),
  diferenças dentro do ruído. As variáveis cruzadas somam 18% da importância do LightGBM, mas
  não se convertem em ganho fora da amostra (sobreajuste).
- Exceções por produto: suavizar ajuda onde a mediana tem ruído de amostragem (Farinha de
  Trigo: Holt com 0,69 do erro do ingênuo; Manteiga: 0,87).

**Conclusão.** No horizonte mensal, os preços da cesta se comportam como um **passeio
aleatório**. O preço atual é a melhor previsão disponível, e os modelos mais complexos sobreajustam
com 30 a 47 meses de histórico. O objetivo foi atingido no sentido de *medir* a
previsibilidade de forma rigorosa, e a resposta é negativa. Limitações: série curta (4 anos,
pouco para estimar sazonalidade) e ausência de variáveis externas (câmbio, clima, preço ao
produtor), que seriam o caminho natural para ganhar poder preditivo.

---

## Questão 2: Algum supermercado antecipa as mudanças de preço dos demais? Com que rapidez os preços convergem?

**Questão.** As variações de preço de uma loja precedem as do restante do mercado (liderança)?
As lojas seguem o mercado com atraso? Quanto tempo leva para um preço "fora da curva" voltar
ao preço de mercado?

**Objetivo e motivação.** Identificar lojas líderes funciona como sinal antecedente de inflação
local. Medir a velocidade de convergência mostra o grau de competição: convergência rápida
indica mercado competitivo, e preços que se mantêm diferentes indicam poder de mercado ou
posicionamento. Isso orienta o consumidor sobre onde e quando comparar preços.

**Método** (`04_q2_mercados.py`), com 28 lojas com ≥ 24 meses e as variações "matched-model":
- **Liderança**, para cada loja *i*, empilhando os 14 produtos:
  `d_resto(t) = a + β·d_i(t−1) + c·d_resto(t−1) + efeito-produto`, onde `d_resto` é a média das
  *demais* lojas (leave-one-out, sem contaminar com a própria loja). Erros-padrão agrupados por
  mês, correção **Benjamini-Hochberg** entre as 28 lojas e **placebo por permutação**
  (deslocamento circular da série da loja, 200 repetições).
- **Seguidor**: `d_i(t) = a + β·d_resto(t−1) + c·d_i(t−1)`.
- **Confundidor checado**: o dia médio de coleta de cada loja. Se a loja é pesquisada mais tarde
  no mês, a variação dela "chega antes" no calendário mensal.
- **Convergência**: `Δr(t) = λ·r(t−1)`, com `r` = log(preço da loja / média das demais), e
  meia-vida = ln(0,5)/ln(1+λ).

**Resultados** (`q2_lideranca_por_loja.csv`, `q2_lideranca_lojas.png`, `q2_meia_vida.png`):
- **Líderes:** só **SUP_43** é significativa após BH (β = 0,10, q = 0,013, p de permutação =
  0,005). SUP_01 tem o maior β (0,18), mas fica no limite (q = 0,067).
- **Porém** o coeficiente de liderança **se correlaciona com o dia médio de coleta**
  (Spearman 0,41, p = 0,03; `q2_lideranca_dia_coleta.png`). SUP_01 e SUP_43 estão entre as lojas
  coletadas mais tarde (dias 18 e 17). Parte da "liderança" é artefato do desenho da coleta.
- **Seguidores:** **16 das 28 lojas** acompanham significativamente o mercado com 1 mês de
  atraso (q < 0,05).
- Modelo empilhado (`q2_modelo_empilhado.csv`): a variação da loja acompanha o mercado no
  próprio mês (0,90), completa o ajuste no mês seguinte (+0,41) e desfaz 42% da própria
  variação anterior (−0,42). As mudanças isoladas de uma loja são majoritariamente temporárias
  (promoções).
- **Meia-vida dos desvios:** cerca de 1 mês (Pão Francês, Tomate, Arroz, Café) até cerca de 3
  meses (Óleo, Açúcar).
- **Posição relativa persistente:** a correlação do nível relativo das lojas entre anos é de
  0,88 a 0,94. Loja cara continua cara.

**Conclusão.** Não há evidência robusta de um supermercado "formador de preço". O único
candidato (SUP_43) fica sob suspeita do efeito do dia de coleta. O padrão dominante é o de um
**mercado de seguidores**: as lojas reagem a choques comuns (atacado, produtor), ajustam-se em
1 a 2 meses, e as diferenças entre lojas são mais de *posicionamento* permanente do que de
timing. Para separar liderança real de artefato de coleta seriam necessários dados semanais ou
com data de coleta padronizada.

---

## Questão 3: O preço de um produto antecipa o de outro?

**Questão.** Na cadeia produtiva, o insumo antecipa o derivado? Hipóteses definidas *a priori*:
**Farinha de Trigo → Pão Francês** e **Leite → Manteiga**. Como triagem exploratória, todos os
182 pares ordenados.

**Objetivo e motivação.** Se um insumo antecipar o derivado, o primeiro funciona como indicador
antecedente. Isso responde à ideia "prever o preço de um produto com base em outros" e
complementa a Q1.

**Método** (`05_q3_produtos.py`), com as variações mensais do índice encadeado (47 meses,
estacionárias):
1. Correlação cruzada com defasagens de −3 a +3.
2. **Causalidade de Granger** (VAR bivariado, defasagem por AIC até 3). Hipóteses a priori com
   α = 5%; triagem com **Benjamini-Hochberg**. Direção reversa testada como controle.
3. **Validação fora da amostra**: ARX (com o outro produto) contra AR (só o próprio), origem
   móvel nos últimos 18 meses, com teste de Diebold-Mariano. Significância estatística dentro da
   amostra não garante utilidade preditiva.

**Resultados** (`q3_granger_todos_pares.csv`, `q3_validacao_fora_amostra.csv`,
`q3_granger_matriz.png`, `q3_ccf_pares.png`):
- **Leite → Manteiga:** p = 0,062 (lag 1, correlação defasada de 0,26), no limite. O reverso
  (Manteiga → Leite) dá p = 0,78, o que é coerente com a direção da cadeia. Fora da amostra, o
  ARX reduz o erro em 5%, sem significância (p = 0,54).
- **Farinha de Trigo → Pão Francês:** p = 0,30, sem evidência. O pão é o produto de preço mais
  rígido (63% dos itens não mudam no mês), então o repasse, se existe, é lento e em degraus.
- **Triagem:** 17 dos 182 pares têm p < 0,05 contra cerca de 9 esperados por acaso, mas
  **nenhum sobrevive à correção BH** (menor q = 0,21). Nenhum par melhora significativamente a
  previsão fora da amostra.
- Poder estatístico: com n = 47, só correlações acima de |r| ≈ 0,40 seriam detectadas com 80%
  de poder.

**Conclusão.** Há um indício fraco e coerente com a cadeia produtiva para Leite → Manteiga, mas
**não há evidência estatística robusta** de que o preço de varejo de um produto antecipe outro
na escala mensal. Isso confirma o achado da Q1 (variáveis cruzadas não melhoram a previsão).
A transmissão de custos provavelmente começa *antes* do varejo. O próximo passo natural é
comparar com o **preço ao produtor** (ex.: CEPEA/Esalq para leite, trigo, café e boi), a ideia 4
do enunciado.
