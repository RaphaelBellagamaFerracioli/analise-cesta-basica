"""Questão 3 - O preço de um produto antecipa o de outro (ex.: insumo -> derivado)?

Hipóteses a priori (cadeia produtiva):  Farinha de Trigo -> Pão Francês;  Leite -> Manteiga.
Triagem exploratória: todos os 14x13 pares, com correção de Benjamini-Hochberg.

Séries: variação mensal "matched-model" (índice de Jevons encadeado) por produto - estacionária.
Métodos:
  1) Correlação cruzada com defasagens 0..3 (IC aproximado ±1,96/sqrt(n)).
  2) Causalidade de Granger (VAR bivariado, defasagem escolhida por AIC até 3).
  3) Validação fora da amostra: ARX (com o outro produto) x AR (só o próprio), origem móvel,
     comparação por MAE e teste de Diebold-Mariano.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import ccf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

from utils import PROC, TAB, CORES, MUDO, TINTA2, PRODUTOS_NUCLEO, DIVERGENTE, estilo, salvar

warnings.filterwarnings("ignore")
estilo()
MAX_LAG = 3
N_TESTE = 18
PARES_A_PRIORI = [("Farinha de Trigo", "Pão Francês"), ("Leite", "Manteiga")]

idx = pd.read_csv(PROC / "indice_produto.csv", parse_dates=["mes"])
R = (idx[idx.produto.isin(PRODUTOS_NUCLEO)]
     .pivot(index="mes", columns="produto", values="dlog_encadeado")[PRODUTOS_NUCLEO]
     .iloc[1:])  # 1º mês não tem variação
print("meses de variação:", len(R), "| NaN:", int(R.isna().sum().sum()))
R = R.fillna(0)


def granger(x, y, maxlag=MAX_LAG):
    """H0: x não Granger-causa y. Retorna (lag escolhido, p-valor F)."""
    dados = pd.concat([y.rename("y"), x.rename("x")], axis=1)
    m = VAR(dados)
    lag = max(1, m.select_order(maxlag).selected_orders["aic"])
    res = m.fit(lag)
    t = res.test_causality("y", ["x"], kind="f")
    return lag, t.pvalue


# ---------------------------------------------------------------- triagem de todos os pares
linhas = []
for x in PRODUTOS_NUCLEO:
    for y in PRODUTOS_NUCLEO:
        if x == y:
            continue
        lag, p = granger(R[x], R[y])
        cc = [R[y].corr(R[x].shift(k)) for k in range(0, MAX_LAG + 1)]
        linhas.append(dict(origem=x, destino=y, lag=lag, p_granger=p,
                           **{f"ccf_lag{k}": c for k, c in enumerate(cc)},
                           a_priori=(x, y) in PARES_A_PRIORI))
gr = pd.DataFrame(linhas)
gr["q_BH"] = multipletests(gr.p_granger, method="fdr_bh")[1]
gr = gr.sort_values("p_granger")
gr.to_csv(TAB / "q3_granger_todos_pares.csv", index=False, float_format="%.4f")
print("Pares com p < 0,05 (sem correção):", int((gr.p_granger < 0.05).sum()), "de", len(gr),
      f"(esperado por acaso ≈ {0.05 * len(gr):.0f})")
print("Pares com q_BH < 0,10:", int((gr.q_BH < 0.10).sum()))
print(gr.head(12).round(3).to_string(index=False))
print("\nHipóteses a priori:")
print(gr[gr.a_priori].round(3).to_string(index=False))
# direção reversa das hipóteses (controle)
for x, y in PARES_A_PRIORI:
    print(f"reverso {y} -> {x}:", gr[(gr.origem == y) & (gr.destino == x)][["lag", "p_granger"]]
          .round(3).to_dict("records"))


# ---------------------------------------------------------------- validação fora da amostra
def avaliar_oos(x, y, lags=MAX_LAG):
    """AR(p) x ARX(p,p) estimados por MQO; p escolhido por AIC no treino de cada origem."""
    df = pd.DataFrame({"y": R[y], "x": R[x]})
    for k in range(1, lags + 1):
        df[f"y{k}"] = df.y.shift(k)
        df[f"x{k}"] = df.x.shift(k)
    df = df.dropna()
    e_ar, e_arx = [], []
    n = len(df)
    for T in range(n - N_TESTE, n):
        tr, te = df.iloc[:T], df.iloc[[T]]
        melhor = {}
        for nome, usa_x in [("ar", False), ("arx", True)]:
            best = (np.inf, None)
            for p in range(1, lags + 1):
                cols = [f"y{k}" for k in range(1, p + 1)] + ([f"x{k}" for k in range(1, p + 1)] if usa_x else [])
                m = LinearRegression().fit(tr[cols], tr.y)
                rss = np.sum((m.predict(tr[cols]) - tr.y) ** 2)
                aic = len(tr) * np.log(rss / len(tr)) + 2 * (len(cols) + 1)
                if aic < best[0]:
                    best = (aic, (m, cols))
            m, cols = best[1]
            melhor[nome] = m.predict(te[cols])[0]
        e_ar.append(abs(melhor["ar"] - te.y.iloc[0]))
        e_arx.append(abs(melhor["arx"] - te.y.iloc[0]))
    e_ar, e_arx = np.array(e_ar), np.array(e_arx)
    d = e_arx - e_ar
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return dict(origem=x, destino=y, MAE_AR=e_ar.mean(), MAE_ARX=e_arx.mean(),
                razao=e_arx.mean() / e_ar.mean(), DM=dm, p_DM=2 * stats.t.sf(abs(dm), len(d) - 1))


pares_oos = PARES_A_PRIORI + [(y, x) for x, y in PARES_A_PRIORI]
pares_oos += [tuple(r) for r in gr[~gr.a_priori].head(5)[["origem", "destino"]].to_numpy()]
oos = pd.DataFrame([avaliar_oos(x, y) for x, y in pares_oos])
oos.to_csv(TAB / "q3_validacao_fora_amostra.csv", index=False, float_format="%.4f")
print("\nValidação fora da amostra (MAE da variação mensal, últimos 18 meses):")
print(oos.round(4).to_string(index=False))

# ---------------------------------------------------------------- gráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, (x, y) in zip(axs, PARES_A_PRIORI):
    n = len(R)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    vals = [R[y].corr(R[x].shift(k)) for k in lags]
    ax.bar(lags, vals, color=[CORES[0] if k > 0 else MUDO for k in lags], width=0.6)
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=MUDO, alpha=0.15, lw=0)
    ax.axhline(0, color=TINTA2, lw=0.8)
    ax.set_xticks(lags)
    ax.set_xlabel(f"defasagem k (meses): corr({y}(t), {x}(t-k))")
    ax.set_title(f"{x} → {y}")
fig.suptitle("Q3 - Correlação cruzada das variações mensais (faixa cinza = IC 95% sob H0)",
             fontweight="bold")
fig.tight_layout()
salvar(fig, "q3_ccf_pares")

M = gr.pivot(index="origem", columns="destino", values="p_granger").loc[PRODUTOS_NUCLEO, PRODUTOS_NUCLEO]
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(-np.log10(M.values), cmap="Blues", vmin=0, vmax=3)
ax.set_xticks(range(len(M)), M.columns, rotation=60, ha="right")
ax.set_yticks(range(len(M)), M.index)
ax.set_xlabel("destino (y)"); ax.set_ylabel("origem (x)")
ax.grid(False)
Q = gr.pivot(index="origem", columns="destino", values="q_BH").loc[PRODUTOS_NUCLEO, PRODUTOS_NUCLEO]
for i in range(len(M)):
    for j in range(len(M)):
        if i != j and M.values[i, j] < 0.05:
            ax.text(j, i, "●" if Q.values[i, j] < 0.10 else "○", ha="center", va="center",
                    fontsize=8, color="white" if M.values[i, j] < 0.01 else TINTA2)
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("-log10(p) Granger")
ax.set_title("Q3 - Causalidade de Granger x → y  (○ p < 0,05;  ● q_BH < 0,10)")
salvar(fig, "q3_granger_matriz")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, (x, y) in zip(axs, PARES_A_PRIORI):
    ix = np.exp(R[[x, y]].cumsum()) * 100
    ax.plot(ix.index, ix[x], color=CORES[0], label=x)
    ax.plot(ix.index, ix[y], color=CORES[1], label=y)
    ax.set_ylabel("índice (out/2022 = 100)")
    ax.legend(loc="upper left")
    ax.set_title(f"{x} e {y}")
fig.tight_layout()
salvar(fig, "q3_indices_pares")
