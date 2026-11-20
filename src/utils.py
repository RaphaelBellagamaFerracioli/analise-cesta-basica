"""Caminhos, paleta e estilo de gráficos compartilhados pelos scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
BRUTO = RAIZ / "dados" / "bruto" / "ICB_DADOS_2026.xlsx"
PROC = RAIZ / "dados" / "processado"
FIG = RAIZ / "resultados" / "figuras"
TAB = RAIZ / "resultados" / "tabelas"
for p in (PROC, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

# Paleta categórica (ordem fixa) e tons neutros
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
MUDO = "#898781"
GRADE = "#e6e5e1"
SUPERFICIE = "#fcfcfb"
AZUL_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGENTE = "RdBu_r"  # azul <-> vermelho, meio cinza-claro

# Produtos com série completa (set/2022 a ago/2026)
PRODUTOS_NUCLEO = [
    "Arroz", "Feijão", "Açúcar", "Café", "Leite", "Manteiga", "Óleo",
    "Farinha de Trigo", "Pão Francês", "Carne Bovina Coxão Mole",
    "Tomate", "Batata", "Banana Prata", "Banana Nanica",
]


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": MUDO,
        "axes.labelcolor": TINTA2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "xtick.color": MUDO,
        "ytick.color": MUDO,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def salvar(fig, nome):
    fig.savefig(FIG / f"{nome}.png")
    plt.close(fig)
