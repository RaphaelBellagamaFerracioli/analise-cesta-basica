"""Questão 2 - Algum supermercado antecipa as mudanças de preço dos demais?
E com que velocidade o preço de uma loja volta ao preço de mercado?

(a) Liderança: para cada loja i, regressão empilhada (14 produtos x meses)
      d_resto(t) = a + b * d_i(t-1) + c * d_resto(t-1) + e
    b > 0 significativo => variações da loja i antecedem as do restante do mercado.
    Erros-padrão agrupados por mês; correção de Benjamini-Hochberg (FDR) entre lojas.
    Teste-placebo por permutação (deslocamento circular da série da loja).
(b) Seguidor: d_i(t) = a + b * d_resto(t-1) + c * d_i(t-1)
(c) Convergência: Δr(t) = λ r(t-1), r = log(p_loja / mediana das demais) => meia-vida.

d = variação "matched-model" (mesmo item, meses consecutivos), sem efeito de troca de marca.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils import PROC, TAB, CORES, MUDO, TINTA2, PRODUTOS_NUCLEO, estilo, salvar

warnings.filterwarnings("ignore")
estilo()
rng = np.random.default_rng(42)
MIN_MESES = 24

var = pd.read_parquet(PROC / "var_loja.parquet")
var = var[var.produto.isin(PRODUTOS_NUCLEO)]
itens = pd.read_parquet(PROC / "itens.parquet")
preco = pd.read_parquet(PROC / "preco_loja.parquet")
preco = preco[preco.produto.isin(PRODUTOS_NUCLEO)]

meses_loja = preco.groupby("loja").mes.nunique()
lojas = meses_loja[meses_loja >= MIN_MESES].index.tolist()
print(f"{len(lojas)} lojas com >= {MIN_MESES} meses")

# soma e contagem por produto-mês para obter a média "das demais lojas" (leave-one-out)
agg = var.groupby(["mes", "produto"]).dlog.agg(["sum", "count"]).reset_index()
var = var.merge(agg, on=["mes", "produto"])
var["d_resto"] = (var["sum"] - var.dlog) / (var["count"] - 1)
var = var[var["count"] > 5].drop(columns=["sum", "count"])

var = var.sort_values(["loja", "produto", "mes"])
g = var.groupby(["loja", "produto"])
var["mes_ant"] = g.mes.shift()
var["d_i_lag"] = g.dlog.shift()
var["d_resto_lag"] = g.d_resto.shift()
var = var[var.mes_ant == var.mes - pd.DateOffset(months=1)].dropna()
var.to_csv(TAB / "q2_painel_variacoes.csv", index=False)


def coef_lider(df):
    m = smf.ols("d_resto ~ d_i_lag + d_resto_lag + C(produto)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df.mes.dt.strftime("%Y%m").astype(int)})
    return m.params["d_i_lag"], m.bse["d_i_lag"], m.pvalues["d_i_lag"], int(m.nobs)


def coef_seguidor(df):
    m = smf.ols("dlog ~ d_resto_lag + d_i_lag + C(produto)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df.mes.dt.strftime("%Y%m").astype(int)})
    return m.params["d_resto_lag"], m.bse["d_resto_lag"], m.pvalues["d_resto_lag"]


# Dia médio de coleta (lojas coletadas mais tarde no mês podem "parecer" líderes)
dia = (itens.dropna(subset=["data_coleta"]).groupby("loja").data_coleta
       .apply(lambda s: s.dt.day.mean()).rename("dia_medio_coleta"))

linhas = []
for loja in lojas:
    df = var[var.loja == loja]
    if df.mes.nunique() < 12:
        continue
    b, se, p, n = coef_lider(df)
    bs, ses, ps = coef_seguidor(df)
    # placebo: desloca circularmente a série da loja por k meses (quebra o alinhamento temporal)
    nulos = []
    meses_u = np.sort(df.mes.unique())
    for _ in range(200):
        k = rng.integers(3, len(meses_u) - 3)
        mapa = dict(zip(meses_u, np.roll(meses_u, k)))
        sh = df[["mes", "produto", "d_i_lag"]].copy()
        sh["mes"] = sh.mes.map(mapa)
        dd = df.drop(columns="d_i_lag").merge(sh, on=["mes", "produto"])
        if len(dd) > 30:
            nulos.append(smf.ols("d_resto ~ d_i_lag + d_resto_lag", data=dd).fit().params["d_i_lag"])
    p_perm = (np.sum(np.abs(nulos) >= abs(b)) + 1) / (len(nulos) + 1)
    linhas.append(dict(loja=loja, n_obs=n, meses=df.mes.nunique(),
                       beta_lider=b, ep_lider=se, p_lider=p, p_permutacao=p_perm,
                       beta_seguidor=bs, ep_seguidor=ses, p_seguidor=ps))
lid = pd.DataFrame(linhas).merge(dia, left_on="loja", right_index=True, how="left")
lid["p_lider_BH"] = multipletests(lid.p_lider, method="fdr_bh")[1]
lid["p_seguidor_BH"] = multipletests(lid.p_seguidor, method="fdr_bh")[1]
lid = lid.sort_values("beta_lider", ascending=False)
lid.to_csv(TAB / "q2_lideranca_por_loja.csv", index=False, float_format="%.4f")
print(lid[["loja", "n_obs", "beta_lider", "p_lider", "p_lider_BH", "p_permutacao",
           "beta_seguidor", "p_seguidor_BH", "dia_medio_coleta"]].round(3).to_string(index=False))

rho, p_rho = stats.spearmanr(lid.beta_lider, lid.dia_medio_coleta, nan_policy="omit")
print(f"Spearman(beta_lider, dia médio de coleta) = {rho:.2f} (p = {p_rho:.3f})")

# Modelo empilhado (todas as lojas): o mercado em geral reage com defasagem?
geral = smf.ols("dlog ~ d_resto + d_resto_lag + d_i_lag + C(produto)", data=var).fit(
    cov_type="cluster", cov_kwds={"groups": var.mes.dt.strftime("%Y%m").astype(int)})
tab_geral = pd.DataFrame({"coef": geral.params, "ep": geral.bse, "p": geral.pvalues}).loc[
    ["d_resto", "d_resto_lag", "d_i_lag"]]
tab_geral.to_csv(TAB / "q2_modelo_empilhado.csv", float_format="%.4f")
print(tab_geral.round(4).to_string())

# (c) Convergência ao preço de mercado: meia-vida do desvio relativo
lp = preco[preco.loja.isin(lojas)].copy()
lp["lp"] = np.log(lp.ppk)
s = lp.groupby(["mes", "produto"]).lp.agg(["sum", "count"]).reset_index()
lp = lp.merge(s, on=["mes", "produto"])
# referência: média do log-preço das demais lojas (leave-one-out)
lp["r"] = lp.lp - (lp["sum"] - lp.lp) / (lp["count"] - 1)
lp = lp.sort_values(["loja", "produto", "mes"])
gg = lp.groupby(["loja", "produto"])
lp["r_lag"] = gg.r.shift()
lp["mes_ant"] = gg.mes.shift()
lp = lp[lp.mes_ant == lp.mes - pd.DateOffset(months=1)].dropna(subset=["r_lag"])
lp["dr"] = lp.r - lp.r_lag
conv = []
for p, d in lp.groupby("produto"):
    m = smf.ols("dr ~ r_lag", data=d).fit(cov_type="cluster",
                                          cov_kwds={"groups": d.mes.dt.strftime("%Y%m").astype(int)})
    lam, se = m.params["r_lag"], m.bse["r_lag"]
    rho_ar = 1 + lam
    conv.append(dict(produto=p, lambda_=lam, ep=se, rho=rho_ar,
                     meia_vida_meses=np.log(0.5) / np.log(rho_ar) if 0 < rho_ar < 1 else np.nan,
                     desvio_medio_abs=d.r.abs().mean(), n=len(d)))
conv = pd.DataFrame(conv).sort_values("meia_vida_meses")
conv.to_csv(TAB / "q2_convergencia.csv", index=False, float_format="%.4f")
print(conv.round(3).to_string(index=False))

# Persistência de posição: lojas caras continuam caras?
pos = lp.groupby(["loja", lp.mes.dt.year]).r.mean().unstack()
corr_anos = pos.corr(method="spearman")
corr_anos.to_csv(TAB / "q2_persistencia_posicao_anos.csv", float_format="%.3f")
print("Correlação de Spearman do nível relativo das lojas entre anos:\n", corr_anos.round(2))

# ---------------------------------------------------------------- gráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
ordem = lid.sort_values("beta_lider").loja.tolist()
for ax, col, ep, pcol, tit in [
        (axs[0], "beta_lider", "ep_lider", "p_lider_BH", "Liderança: d_i(t-1) → d_resto(t)"),
        (axs[1], "beta_seguidor", "ep_seguidor", "p_seguidor_BH", "Seguidor: d_resto(t-1) → d_i(t)")]:
    dd = lid.set_index("loja").loc[ordem]
    y = np.arange(len(dd))
    sig = dd[pcol] < 0.05
    ax.errorbar(dd[col], y, xerr=1.96 * dd[ep], fmt="none", ecolor=MUDO, elinewidth=1)
    ax.scatter(dd[col][~sig], y[~sig], s=30, color=MUDO, zorder=3, label="não significativo (BH)")
    ax.scatter(dd[col][sig], y[sig], s=40, color=CORES[0], zorder=3, label="q < 0,05 (BH)")
    ax.axvline(0, color=TINTA2, lw=0.8)
    ax.set_yticks(y, dd.index)
    ax.set_title(tit)
    ax.set_xlabel("coeficiente (IC 95%)")
    ax.grid(axis="y", visible=False)
axs[0].legend(loc="lower right")
fig.suptitle("Q2 - Algum supermercado antecipa as variações de preço do restante do mercado?",
             fontweight="bold")
fig.tight_layout()
salvar(fig, "q2_lideranca_lojas")

fig, ax = plt.subplots(figsize=(7, 5))
c = conv.dropna(subset=["meia_vida_meses"]).sort_values("meia_vida_meses", ascending=False)
ax.barh(c.produto, c.meia_vida_meses, color=CORES[0], height=0.6)
for i, v in enumerate(c.meia_vida_meses):
    ax.text(v + 0.03, i, f"{v:.1f}", va="center", fontsize=8, color=TINTA2)
ax.set_xlabel("meia-vida do desvio em relação às demais lojas (meses)")
ax.set_title("Q2 - Velocidade de convergência ao preço de mercado")
ax.grid(axis="y", visible=False)
salvar(fig, "q2_meia_vida")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(lid.dia_medio_coleta, lid.beta_lider, s=30, color=CORES[0])
for _, r in lid.iterrows():
    ax.annotate(r.loja.replace("SUP_", ""), (r.dia_medio_coleta, r.beta_lider), fontsize=6,
                color=MUDO, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("dia médio de coleta no mês")
ax.set_ylabel("coeficiente de liderança")
ax.set_title(f"Liderança x dia de coleta (Spearman = {rho:.2f}, p = {p_rho:.2f})")
salvar(fig, "q2_lideranca_dia_coleta")
