"""Etapa 2 - análise exploratória: séries, dispersão entre lojas, sazonalidade, correlações."""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

from utils import PROC, TAB, CORES, TINTA2, MUDO, PRODUTOS_NUCLEO, DIVERGENTE, estilo, salvar

warnings.filterwarnings("ignore")
estilo()
idx = pd.read_csv(PROC / "indice_produto.csv", parse_dates=["mes"])
preco_loja = pd.read_parquet(PROC / "preco_loja.parquet")
idx = idx[idx.produto.isin(PRODUTOS_NUCLEO)]

# 1) Séries por produto: mediana simples x índice encadeado (matched-model)
fig, axs = plt.subplots(4, 4, figsize=(14, 11), sharex=True)
for ax, p in zip(axs.flat, PRODUTOS_NUCLEO):
    g = idx[idx.produto == p]
    ax.plot(g.mes, g.ppk_mediana, color=CORES[0], lw=1.6, label="Mediana entre lojas")
    ax.plot(g.mes, g.ppk_encadeado, color=CORES[1], lw=1.6, label="Índice encadeado (mesmo item)")
    ax.set_title(p, fontsize=9)
    ax.set_ylabel("R$/kg" if p not in ("Leite", "Óleo") else "R$/L")
for ax in axs.flat[len(PRODUTOS_NUCLEO):]:
    ax.axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
axs[3, 2].legend(h, l, loc="center", fontsize=10)
fig.suptitle("Preço por produto, set/2022 a ago/2026", fontweight="bold", y=0.995)
fig.tight_layout()
fig.autofmt_xdate()
salvar(fig, "eda_series_produtos")

# 2) Variação acumulada no período
w = idx.pivot(index="mes", columns="produto", values="ppk_mediana")[PRODUTOS_NUCLEO]
var_total = (w.iloc[-3:].mean() / w.iloc[:3].mean() - 1).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
cores = [CORES[0] if v >= 0 else CORES[7] for v in var_total]
ax.barh(var_total.index, var_total.values * 100, color=cores, height=0.6)
for i, v in enumerate(var_total.values * 100):
    ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center",
            ha="left" if v >= 0 else "right", color=TINTA2, fontsize=8)
ax.axvline(0, color=MUDO, lw=0.8)
ax.set_xlabel("Variação do preço mediano (média dos 3 últimos vs 3 primeiros meses), %")
ax.set_title("Variação de preço por produto no período")
ax.grid(axis="y", visible=False)
salvar(fig, "eda_variacao_acumulada")
var_total.rename("variacao").to_csv(TAB / "eda_variacao_acumulada.csv")

# 3) Dispersão entre lojas (coeficiente de variação)
cv = (preco_loja[preco_loja.produto.isin(PRODUTOS_NUCLEO)]
      .groupby(["mes", "produto"]).ppk.agg(lambda s: s.std() / s.mean()).unstack())
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cv.index, cv.median(axis=1) * 100, color=CORES[0])
ax.fill_between(cv.index, cv.quantile(0.25, axis=1) * 100, cv.quantile(0.75, axis=1) * 100,
                color=CORES[0], alpha=0.18, lw=0)
ax.set_ylabel("Coef. de variação entre lojas, %")
ax.set_title("Dispersão de preços entre supermercados (mediana e IQR entre produtos)")
salvar(fig, "eda_dispersao_lojas")
cv.mean().sort_values().rename("cv_medio").to_csv(TAB / "eda_cv_medio_produto.csv")

# 4) Retornos mensais: estacionariedade, sazonalidade, correlação
lw = np.log(w)
r = lw.diff().dropna()
testes = []
for p in PRODUTOS_NUCLEO:
    adf_n = adfuller(lw[p], autolag="AIC")[1]
    kpss_n = kpss(lw[p], regression="c", nlags="auto")[1]
    adf_d = adfuller(r[p], autolag="AIC")[1]
    kpss_d = kpss(r[p], regression="c", nlags="auto")[1]
    testes.append(dict(produto=p, adf_p_nivel=adf_n, kpss_p_nivel=kpss_n,
                       adf_p_diff=adf_d, kpss_p_diff=kpss_d))
testes = pd.DataFrame(testes)
testes.to_csv(TAB / "eda_estacionariedade.csv", index=False, float_format="%.3f")
print(testes.round(3).to_string(index=False))

saz = r.groupby(r.index.month).mean() * 100
fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(saz.T.values, cmap=DIVERGENTE, vmin=-8, vmax=8, aspect="auto")
ax.set_yticks(range(len(saz.columns)), saz.columns)
ax.set_xticks(range(12), ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"])
ax.grid(False)
cb = fig.colorbar(im, ax=ax); cb.set_label("variação média mensal, %")
ax.set_title("Perfil sazonal: variação média por mês do ano (4 anos)")
salvar(fig, "eda_sazonalidade")

corr = r.corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap=DIVERGENTE, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        if i != j and abs(corr.iloc[i, j]) >= 0.3:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlação contemporânea das variações mensais (log)")
salvar(fig, "eda_correlacao_retornos")

# Cesta (soma dos preços medianos, 1 unidade de cada) — indicador-resumo
cesta = w.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(cesta.index, cesta.values, color=CORES[0])
ax.set_ylabel("R$")
ax.set_title("Soma dos preços medianos dos 14 itens (1 kg/L de cada)")
salvar(fig, "eda_cesta")

# Concordância entre mediana e índice encadeado
conc = []
for p in PRODUTOS_NUCLEO:
    g = idx[idx.produto == p]
    a, b = np.log(g.ppk_mediana).diff(), np.log(g.ppk_encadeado).diff()
    conc.append(dict(produto=p, corr_variacoes=a.corr(b),
                     desvpad_var_mediana=a.std(), desvpad_var_encadeado=b.std()))
pd.DataFrame(conc).to_csv(TAB / "eda_mediana_vs_encadeado.csv", index=False, float_format="%.3f")
print(pd.DataFrame(conc).round(3).to_string(index=False))
