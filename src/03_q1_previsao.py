"""Questão 1 - É possível prever o preço mediano dos itens da cesta 1 e 3 meses à frente
melhor do que o passeio aleatório? Informação de OUTROS produtos ajuda?

Validação: origem móvel com janela expansível (rolling-origin), últimos 18 meses como teste.
Todos os modelos trabalham em log-preço; erros avaliados em R$.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from utils import PROC, TAB, CORES, MUDO, TINTA2, PRODUTOS_NUCLEO, estilo, salvar

warnings.filterwarnings("ignore")
estilo()

N_TESTE = 18
HORIZONTES = [1, 3]
VAL_INTERNA = 6  # meses finais do treino usados para escolher hiperparâmetros dos modelos de ML

idx = pd.read_csv(PROC / "indice_produto.csv", parse_dates=["mes"])
W = (idx[idx.produto.isin(PRODUTOS_NUCLEO)]
     .pivot(index="mes", columns="produto", values="ppk_mediana")[PRODUTOS_NUCLEO])
Y = np.log(W)
T_TOTAL = len(Y)
MESES = Y.index
PRIMEIRA_ORIGEM = T_TOTAL - N_TESTE - 1  # origem cujo h=1 é o 1º mês de teste


# ---------------------------------------------------------------- modelos univariados
def prev_naive(y, h):
    return y[-1]


def prev_snaive(y, h):
    return y[len(y) - 12 + h - 1] if h <= 12 else y[-1]


def prev_drift(y, h):
    return y[-1] + h * (y[-1] - y[0]) / (len(y) - 1)


def prev_ses(y, h):
    return SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(h)[-1]


def prev_holt_amort(y, h):
    m = ExponentialSmoothing(y, trend="add", damped_trend=True,
                             initialization_method="estimated").fit()
    return m.forecast(h)[-1]


def prev_arima(y, h):
    """ARIMA(p,1,q), p,q em {0,1,2}, escolhido por AICc em cada origem."""
    melhor, melhor_aicc = None, np.inf
    for p in range(3):
        for q in range(3):
            try:
                m = ARIMA(y, order=(p, 1, q)).fit()
            except Exception:
                continue
            if m.aicc < melhor_aicc:
                melhor, melhor_aicc = m, m.aicc
    return melhor.forecast(h)[-1]


UNIVARIADOS = {"Ingênuo (passeio aleatório)": prev_naive, "Ingênuo sazonal": prev_snaive,
               "Drift": prev_drift, "SES": prev_ses, "Holt amortecido": prev_holt_amort,
               "ARIMA (AICc)": prev_arima}


# ---------------------------------------------------------------- modelos globais de ML
def montar_features(Y, h, cruzado):
    """Uma linha por (produto, t). Alvo: y[t+h] - y[t]. Usa somente informação até t."""
    R = Y.diff()
    linhas = []
    for k, p in enumerate(Y.columns):
        y, r = Y[p], R[p]
        f = pd.DataFrame({
            "r0": r, "r1": r.shift(1), "r2": r.shift(2),
            "desvio_media12": y - y.rolling(12).mean(),
            "vol6": r.rolling(6).std(),
            # variação do mesmo período no ano anterior (sazonalidade)
            "saz_ano_ant": y.shift(12 - h) - y.shift(12),
        })
        mes_alvo = (Y.index + pd.DateOffset(months=h)).month
        f["mes_sin"], f["mes_cos"] = np.sin(2 * np.pi * mes_alvo / 12), np.cos(2 * np.pi * mes_alvo / 12)
        f["cesta_r0"] = R.mean(axis=1)
        if cruzado:
            for q in Y.columns:  # retornos (t e t-1) de todos os 14 produtos
                f[f"x0_{q}"] = R[q]
                f[f"x1_{q}"] = R[q].shift(1)
        for q in Y.columns:
            f[f"prod_{q}"] = float(q == p)
        f["alvo"] = y.shift(-h) - y
        f["t"] = np.arange(len(Y))
        f["produto"] = p
        linhas.append(f)
    return pd.concat(linhas).dropna(subset=["r2", "desvio_media12", "vol6", "saz_ano_ant"])


def ajustar_ridge(Xtr, ytr, alpha):
    sc = StandardScaler().fit(Xtr)
    return sc, Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)


GRADE_RIDGE = [0.1, 1, 10, 100, 1000]
GRADE_LGB = [dict(num_leaves=4, n_estimators=150), dict(num_leaves=4, n_estimators=400),
             dict(num_leaves=8, n_estimators=150), dict(num_leaves=8, n_estimators=400)]
LGB_FIXO = dict(learning_rate=0.03, min_child_samples=15, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)


def prever_global(F, T, h, tipo):
    """Treina com linhas cujo alvo já é conhecido em T (t + h <= T) e prevê as linhas t = T."""
    cols = [c for c in F.columns if c not in ("alvo", "t", "produto")]
    tr = F[(F.t + h <= T)]
    te = F[F.t == T]
    # validação interna temporal para hiperparâmetros
    corte = T - h - VAL_INTERNA
    a, v = tr[tr.t <= corte], tr[tr.t > corte]
    if tipo == "ridge":
        erros = []
        for al in GRADE_RIDGE:
            sc, m = ajustar_ridge(a[cols], a.alvo, al)
            erros.append(np.abs(m.predict(sc.transform(v[cols])) - v.alvo).mean())
        al = GRADE_RIDGE[int(np.argmin(erros))]
        sc, m = ajustar_ridge(tr[cols], tr.alvo, al)
        pred = m.predict(sc.transform(te[cols]))
    else:
        erros = []
        for g in GRADE_LGB:
            m = lgb.LGBMRegressor(**g, **LGB_FIXO).fit(a[cols], a.alvo)
            erros.append(np.abs(m.predict(v[cols]) - v.alvo).mean())
        g = GRADE_LGB[int(np.argmin(erros))]
        m = lgb.LGBMRegressor(**g, **LGB_FIXO).fit(tr[cols], tr.alvo)
        pred = m.predict(te[cols])
    return dict(zip(te.produto, pred)), m, cols


# ---------------------------------------------------------------- loop de avaliação
registros = []
importancias = []
for h in HORIZONTES:
    F_proprio = montar_features(Y, h, cruzado=False)
    F_cruzado = montar_features(Y, h, cruzado=True)
    for T in range(PRIMEIRA_ORIGEM, T_TOTAL - h):
        alvo_mes = MESES[T + h]
        prevs = {}
        for p in PRODUTOS_NUCLEO:
            y = Y[p].to_numpy()[: T + 1]
            for nome, f in UNIVARIADOS.items():
                prevs[(nome, p)] = f(y, h)
        for nome, F, tipo in [("Ridge (só o próprio produto)", F_proprio, "ridge"),
                              ("Ridge (+ outros produtos)", F_cruzado, "ridge"),
                              ("LightGBM (só o próprio produto)", F_proprio, "lgb"),
                              ("LightGBM (+ outros produtos)", F_cruzado, "lgb")]:
            dp, modelo, cols = prever_global(F, T, h, tipo)
            for p, v in dp.items():
                prevs[(nome, p)] = Y[p].iloc[T] + v
            if tipo == "lgb" and "outros" in nome:
                importancias.append(pd.Series(modelo.booster_.feature_importance("gain"),
                                              index=cols, name=f"h{h}_T{T}"))
        for p in PRODUTOS_NUCLEO:
            prevs[("Combinação (Holt+ARIMA+Ridge)", p)] = np.mean(
                [prevs[("Holt amortecido", p)], prevs[("ARIMA (AICc)", p)],
                 prevs[("Ridge (só o próprio produto)", p)]])
        for (nome, p), v in prevs.items():
            registros.append(dict(h=h, origem=MESES[T], mes_alvo=alvo_mes, modelo=nome, produto=p,
                                  previsto=np.exp(v), real=W[p].iloc[T + h]))
    print(f"h={h} concluído")

res = pd.DataFrame(registros)
res["erro"] = res.previsto - res.real
res["ea"] = res.erro.abs()
res["epa"] = res.ea / res.real * 100
res.to_csv(TAB / "q1_previsoes.csv", index=False)

# ---------------------------------------------------------------- métricas
# Escala MASE: EAM do ingênuo de 1 passo no período de treino inicial (fixa, sem vazamento)
escala = (W.iloc[: PRIMEIRA_ORIGEM + 1].diff().abs().mean())
res["ea_escalado"] = res.ea / res.produto.map(escala)

base = res[res.modelo == "Ingênuo (passeio aleatório)"].set_index(["h", "mes_alvo", "produto"]).ea
res["ea_naive"] = res.set_index(["h", "mes_alvo", "produto"]).index.map(base)

linhas = []
for (h, m), g in res.groupby(["h", "modelo"]):
    rel = g.groupby("produto").ea.mean() / g.groupby("produto").ea_naive.mean()
    linhas.append(dict(h=h, modelo=m, MAPE=g.epa.mean(), MASE=g.ea_escalado.mean(),
                       AvgRelMAE=np.exp(np.log(rel).mean()),
                       produtos_melhor_que_ingenuo=int((rel < 1).sum())))
met = pd.DataFrame(linhas)


def diebold_mariano(d, h):
    """Teste DM com correção de Harvey-Leybourne-Newbold; d = perda(modelo) - perda(ingênuo)."""
    n = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), df=n - 1)


dms = []
for (h, m), g in res.groupby(["h", "modelo"]):
    # perda: erro percentual absoluto médio entre produtos em cada mês-alvo (livre de escala)
    nv = res[(res.h == h) & (res.modelo == "Ingênuo (passeio aleatório)")].groupby("mes_alvo").epa.mean()
    d = (g.groupby("mes_alvo").epa.mean() - nv).to_numpy()
    dm, pv = diebold_mariano(d, h) if m != "Ingênuo (passeio aleatório)" else (np.nan, np.nan)
    dms.append(dict(h=h, modelo=m, DM=dm, p_valor_DM=pv))
met = met.merge(pd.DataFrame(dms), on=["h", "modelo"]).sort_values(["h", "AvgRelMAE"])
met.to_csv(TAB / "q1_metricas.csv", index=False, float_format="%.4f")
print(met.round(3).to_string(index=False))

# Erro relativo ao ingênuo por produto (h=1 e h=3)
rel_prod = (res.groupby(["h", "modelo", "produto"]).ea.mean()
            / res.groupby(["h", "modelo", "produto"]).ea_naive.mean()).unstack("produto")
rel_prod.to_csv(TAB / "q1_relMAE_por_produto.csv", float_format="%.3f")

# Importância de variáveis (LightGBM + outros produtos)
imp = pd.concat(importancias, axis=1).mean(axis=1)
imp = (imp / imp.sum()).sort_values(ascending=False)
imp.rename("ganho_relativo").to_csv(TAB / "q1_importancia_lgb_cruzado.csv")
grupo = imp.groupby(lambda c: "outros produtos (lags)" if c.startswith("x") else
                    ("identificador do produto" if c.startswith("prod_") else c)).sum()
print(grupo.sort_values(ascending=False).round(3).to_string())

# ---------------------------------------------------------------- gráficos
ordem = met[met.h == 1].sort_values("AvgRelMAE").modelo.tolist()
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, h in zip(axs, HORIZONTES):
    mm = met[met.h == h].set_index("modelo").loc[ordem[::-1]]
    cores = [CORES[0] if v < 1 else MUDO for v in mm.AvgRelMAE]
    ax.barh(mm.index, mm.AvgRelMAE, color=cores, height=0.6)
    ax.axvline(1, color=CORES[7], lw=1.2, ls="--")
    for i, (v, pv) in enumerate(zip(mm.AvgRelMAE, mm.p_valor_DM)):
        marca = "*" if pd.notna(pv) and pv < 0.05 else ""
        ax.text(v + 0.01, i, f"{v:.2f}{marca}", va="center", fontsize=8, color=TINTA2)
    ax.set_title(f"h = {h} {'mês' if h == 1 else 'meses'}")
    ax.set_xlabel("AvgRelMAE (erro relativo ao passeio aleatório; < 1 = melhor)")
    ax.grid(axis="y", visible=False)
fig.suptitle("Q1 - Desempenho fora da amostra (18 meses, 14 produtos). * = diferença significativa vs passeio aleatório (Diebold-Mariano, p < 0,05)",
             fontweight="bold")
fig.tight_layout()
salvar(fig, "q1_avgrelmae")

fig, ax = plt.subplots(figsize=(10, 6))
r1 = rel_prod.loc[1].loc[[m for m in ordem if m != "Ingênuo (passeio aleatório)"]]
im = ax.imshow(np.log2(r1.values), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(len(r1)), r1.index)
ax.set_xticks(range(r1.shape[1]), r1.columns, rotation=55, ha="right")
ax.grid(False)
for i in range(r1.shape[0]):
    for j in range(r1.shape[1]):
        ax.text(j, i, f"{r1.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
cb = fig.colorbar(im, ax=ax); cb.set_label("log2(MAE modelo / MAE ingênuo)  —  azul = melhor")
ax.set_title("Q1 - MAE relativo ao passeio aleatório por produto (h = 1)")
salvar(fig, "q1_relmae_produtos")

# Previsto x real para alguns produtos (h=1)
sel = ["Café", "Arroz", "Leite", "Tomate"]
mods = ["Ingênuo (passeio aleatório)", "ARIMA (AICc)", "Ridge (só o próprio produto)"]
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for ax, p in zip(axs.flat, sel):
    ax.plot(W.index, W[p], color=MUDO, lw=1.4, label="Real")
    for c, m in zip(CORES, mods):
        g = res[(res.h == 1) & (res.modelo == m) & (res.produto == p)]
        ax.plot(g.mes_alvo, g.previsto, color=c, lw=1.6, label=m)
    ax.axvline(MESES[PRIMEIRA_ORIGEM + 1], color=MUDO, ls=":", lw=1)
    ax.set_title(p)
axs[0, 0].legend(loc="upper left")
fig.suptitle("Q1 - Previsões 1 mês à frente no período de teste", fontweight="bold")
fig.tight_layout()
salvar(fig, "q1_previsto_real")
