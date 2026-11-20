"""Etapa 1 - limpeza, harmonização e construção dos índices de preço.

Saídas (dados/processado):
  itens.parquet         observações limpas (1 linha = 1 item coletado)
  preco_loja.parquet    preço por kg (mediana entre marcas) por mês x loja x produto
  var_loja.parquet      variação mensal "matched-model" por loja x produto
  indice_produto.csv    índice mensal por produto (mediana simples e encadeado)
"""
import numpy as np
import pandas as pd

from utils import BRUTO, PROC, TAB, PRODUTOS_NUCLEO

LIMIAR_OUTLIER = np.log(2)  # descarta PPK fora de [mediana/2, 2*mediana] no mês-produto

# Em mar/2026 a planilha passou a usar nomes abreviados; unificamos.
MAPA_NOMES = {
    "Carne Acem": "Carne Bovina Acem",
    "Carne Coxão Mole": "Carne Bovina Coxão Mole",
    "Carne Pernil": "Carne Suína Pernil",
    "Pernil": "Carne Suína Pernil",
    "Macarrão": "Macarrão com Ovos",
    "Pão": "Pão Francês",
    "Farinha": "Farinha de Trigo",
}
# Cortes com 1-2 meses de coleta: sem série temporal possível
DESCARTAR = {"Carne Bovina Alcatra", "Carne Bovina Patinho", "Carne Bovina Músculo"}


def main():
    bruto = pd.read_excel(BRUTO)
    log = [("linhas brutas", len(bruto))]

    d = bruto.rename(columns={"Mes_Ano_Data": "mes", "Data_Coleta": "data_coleta",
                              "Estab": "loja", "Produto": "produto", "Marca": "marca",
                              "Preco": "preco", "Quantidade": "qtd", "PPK": "ppk"})
    # Linhas idênticas em todas as colunas (923 só em dez/2024) são duplicação de digitação
    d = d.drop_duplicates()
    log.append(("após remover duplicatas exatas", len(d)))
    # Em dez/2024 a marca de itens sem marca foi registrada como o número 0
    d["marca"] = d["marca"].where(d["marca"].map(lambda v: isinstance(v, str)), np.nan)

    d["produto"] = d["produto"].replace(MAPA_NOMES)
    d = d[~d["produto"].isin(DESCARTAR)]
    log.append(("após remover cortes sem série", len(d)))

    # Estabelecimentos "ALT_" aparecem num único mês (substitutos pontuais)
    d = d[~d["loja"].str.startswith("ALT")]
    log.append(("após remover estabelecimentos ALT (1 mês)", len(d)))

    d = d[(d["ppk"] > 0) & d["ppk"].notna()]
    med = d.groupby(["mes", "produto"])["ppk"].transform("median")
    desvio = np.log(d["ppk"] / med)
    outl = desvio.abs() > LIMIAR_OUTLIER
    d[outl].to_csv(TAB / "prep_outliers_removidos.csv", index=False)
    d = d[~outl].copy()
    log.append(("após filtro de outliers (|log(ppk/mediana)| > ln 2)", len(d)))

    d["marca"] = d["marca"].fillna("(sem marca)").str.strip().str.title()
    # chave de item: mesma loja, produto, marca e embalagem
    d["item"] = d["loja"] + "|" + d["produto"] + "|" + d["marca"] + "|" + d["qtd"].astype(str)
    d.to_parquet(PROC / "itens.parquet")

    # Preço da loja = mediana entre os itens (marcas/embalagens) coletados
    preco_loja = (d.groupby(["mes", "loja", "produto"])["ppk"].median()
                  .rename("ppk").reset_index())
    preco_loja.to_parquet(PROC / "preco_loja.parquet")

    # Variação "matched-model": compara apenas o MESMO item em meses consecutivos,
    # eliminando o efeito de troca de marca/embalagem na mediana.
    it = d.groupby(["item", "loja", "produto", "mes"])["ppk"].median().reset_index()
    it = it.sort_values(["item", "mes"])
    it["mes_ant"] = it.groupby("item")["mes"].shift()
    it["ppk_ant"] = it.groupby("item")["ppk"].shift()
    consecutivo = it["mes_ant"] == it["mes"] - pd.DateOffset(months=1)
    it = it[consecutivo].copy()
    it["dlog"] = np.log(it["ppk"] / it["ppk_ant"])
    # Rigidez de preços: fração de itens cujo preço não mudou de um mês ao outro
    it["sem_mudanca"] = np.isclose(it["ppk"], it["ppk_ant"])
    (it.groupby("produto")["sem_mudanca"].mean().sort_values()
     .to_csv(TAB / "prep_rigidez_precos.csv"))
    # Média dos log-retornos (índice de Jevons). A mediana seria enviesada para 0,
    # pois 13% a 63% dos itens não mudam de preço a cada mês.
    var_loja = (it.groupby(["mes", "loja", "produto"])["dlog"].mean()
                .rename("dlog").reset_index())
    var_loja.to_parquet(PROC / "var_loja.parquet")

    # Índice por produto
    mediana = preco_loja.groupby(["mes", "produto"]).agg(
        ppk_mediana=("ppk", "median"), n_lojas=("loja", "nunique"))
    dmed = var_loja.groupby(["mes", "produto"])["dlog"].mean().rename("dlog_encadeado")
    idx = mediana.join(dmed, how="left").reset_index().sort_values(["produto", "mes"])
    # Índice encadeado, ancorado no nível mediano do 1º mês do produto (R$/kg)
    partes = []
    for p, g in idx.groupby("produto"):
        g = g.copy()
        dl = g["dlog_encadeado"].fillna(0).to_numpy(copy=True)
        dl[0] = 0
        g["ppk_encadeado"] = g["ppk_mediana"].iloc[0] * np.exp(np.cumsum(dl))
        partes.append(g)
    idx = pd.concat(partes)
    idx.to_csv(PROC / "indice_produto.csv", index=False)

    # Relatório de preparação
    pd.DataFrame(log, columns=["etapa", "linhas"]).to_csv(TAB / "prep_funil.csv", index=False)
    cobertura = idx.groupby("produto").agg(inicio=("mes", "min"), fim=("mes", "max"),
                                           meses=("mes", "nunique"),
                                           lojas_mediana=("n_lojas", "median"))
    cobertura.to_csv(TAB / "prep_cobertura_produtos.csv")
    print(pd.DataFrame(log, columns=["etapa", "linhas"]).to_string(index=False))
    print(cobertura.to_string())
    faltando = [p for p in PRODUTOS_NUCLEO if cobertura.loc[p, "meses"] < 48]
    print("núcleo com meses faltando:", faltando)

if __name__ == "__main__":
    main()
