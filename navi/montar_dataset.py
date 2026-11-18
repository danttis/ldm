#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
montar_dataset.py — Constrói o corpus de treino para a LLM de fretes.

Fontes usadas:
  dados/comex/imp/IMP_YYYY.csv   → frete declarado (VL_FRETE) de importações marítimas
  dados/comex/aux/TABELAS_AUXILIARES.xlsx → decodificação de NCM e país
  dados/indices/brent_diario.csv  → preço do Brent (mensal) como covariável
  dados/unctad/fretes_maritimos_origem_brasil.csv → fretes de exportação (UNCTAD GTCDIT)
  dados/noticias/*.parquet|csv    → artigos de navegação/petróleo (contexto de linguagem)

Saída:
  treino/corpus/train.txt / val.txt   (uma amostra por linha)
  treino/corpus/tokenizer.json        (BPE treinado do zero)
Uso:
  python montar_dataset.py [--n-frete 400000] [--n-noticias 40000] [--seed 42]
"""
import argparse
import csv
import glob
import io
import json
import os
import random
import sys

import numpy as np
import pandas as pd

# Caminhos padrão relativos ao CWD — rode da raiz do projeto, ou passe
# --dados/--out. main() sobrescreve com os valores dos argumentos.
DADOS = "dados"
COMEX = os.path.join(DADOS, "comex")
OUT = os.path.join("treino", "corpus")

# ---------------------------------------------------------------- decodificadores

def carregar_decodificadores():
    """Lê TABELAS_AUXILIARES.xlsx → dicionários NCM (aba 1) e PAIS (aba 10)."""
    xf = os.path.join(COMEX, "aux", "TABELAS_AUXILIARES.xlsx")
    ncm, pais = {}, {}
    d1 = pd.read_excel(xf, sheet_name="1", dtype=str)
    for _, r in d1.iterrows():
        ncm[str(r["CO_NCM"]).zfill(8)] = str(r["NO_NCM_POR"])[:80]
    d10 = pd.read_excel(xf, sheet_name="10", dtype=str)
    for _, r in d10.iterrows():
        pais[str(r["CO_PAIS"]).zfill(3)] = str(r["NO_PAIS"])[:50]
    print(f"decodificadores: {len(ncm)} NCMs, {len(pais)} países")
    return ncm, pais


def carregar_brent_mensal():
    f = os.path.join(DADOS, "indices", "brent_diario.csv")
    df = pd.read_csv(f)
    df.columns = ["data", "preco"]
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df["data"] = pd.to_datetime(df["data"])
    df["ano"], df["mes"] = df["data"].dt.year, df["data"].dt.month
    m = df.groupby(["ano", "mes"])["preco"].mean().round(1)
    return m.to_dict()

# ---------------------------------------------------------------- templates

def fmt_num(x, casas=0):
    try:
        return f"{float(x):,.{casas}f}".replace(",", "_")
    except Exception:
        return "NA"


def linha_frete(ano, mes, ncm, ncm_nome, origem, uf, peso_kg, fob, brent, frete, frete_kg):
    """Template 1: o alvo (frete) vem NO FIM — o modelo aprende a completá-lo."""
    return (
        f"FRETE maritimo importacao | ano={ano} | mes={mes:02d} "
        f"| produto={ncm_nome} NCM={ncm} | origem={origem} | uf={uf} "
        f"| peso_kg={fmt_num(peso_kg)} | fob_usd={fmt_num(fob)} | brent={brent} "
        f"=> frete_usd={fmt_num(frete)} | frete_usd_por_kg={frete_kg}"
    )


def linha_frete_export(ano, produto, destino, custo_fob, usd_kg):
    """Template 2: fretes de exportação Brasil→mundo (UNCTAD GTCDIT)."""
    return (
        f"FRETE maritimo exportacao Brasil | ano={ano} "
        f"| produto={produto} | destino={destino} "
        f"| custo_pct_fob={custo_fob} => frete_usd_por_kg={usd_kg}"
    )

# ---------------------------------------------------------------- montagem

def amostras_comex(ncm_dic, pais_dic, brent_m, n_max, seed):
    arqs = sorted(glob.glob(os.path.join(COMEX, "imp", "IMP_*.csv")))[-11:]  # últimos 11 anos
    rng = random.Random(seed)
    out = []
    frac = min(1.0, n_max / 6_000_000 + 0.02)  # amostragem probabilística p/ caber na RAM
    for arq in arqs:
        try:
            df = pd.read_csv(arq, sep=";", encoding="latin-1", usecols=[
                "CO_ANO", "CO_MES", "CO_NCM", "CO_PAIS", "SG_UF_NCM", "CO_VIA",
                "KG_LIQUIDO", "VL_FOB", "VL_FRETE"], dtype=str)
        except Exception as e:
            print(f"  pulando {os.path.basename(arq)}: {e}"); continue
        df = df[df["CO_VIA"].str.strip() == "01"]        # marítimo
        df = df[pd.to_numeric(df["VL_FRETE"], errors="coerce") > 0]
        df = df[pd.to_numeric(df["KG_LIQUIDO"], errors="coerce") > 0]
        df = df.sample(frac=min(1.0, frac), random_state=seed)
        for r in df.itertuples(index=False):
            try:
                peso = float(r.KG_LIQUIDO); fob = float(r.VL_FOB); frete = float(r.VL_FRETE)
                if peso <= 0 or fob <= 0:
                    continue
                fk = f"{frete / peso:.4f}"                 # alvo principal
                b = brent_m.get((int(r.CO_ANO), int(r.CO_MES)), "NA")
                nome = ncm_dic.get(str(r.CO_NCM).zfill(8), "produto NCM " + str(r.CO_NCM))
                origem = pais_dic.get(str(r.CO_PAIS).zfill(3), str(r.CO_PAIS))
                out.append(linha_frete(int(r.CO_ANO), int(r.CO_MES), str(r.CO_NCM),
                                       nome, origem, r.SG_UF_NCM, peso, fob, b, frete, fk))
            except Exception:
                continue
        print(f"  {os.path.basename(arq)}: {len(out)} amostras acumuladas")
        if len(out) >= n_max:
            break
    return out[:n_max]


def amostras_unctad(n_max, seed):
    f = os.path.join(DADOS, "unctad", "fretes_maritimos_origem_brasil.csv")
    if not os.path.exists(f):
        return []
    df = pd.read_csv(f, usecols=["Year", "HS2012Product Label", "Destination Label",
                                 "Transport costs to FOB value", "Transport costs per unit"])
    df = df[["Year", "HS2012Product Label", "Destination Label",
             "Transport costs to FOB value", "Transport costs per unit"]]  # ordem explícita
    df = df.dropna(subset=["Transport costs per unit", "Transport costs to FOB value"])
    if len(df) > n_max:
        df = df.sample(n_max, random_state=seed)
    out = []
    for r in df.itertuples(index=False):
        try:
            pct = round(float(r._3), 2)          # Transport costs to FOB value
            uk = f"{float(r._4):.4f}"            # Transport costs per unit (US$/kg)
            out.append(linha_frete_export(int(r.Year), str(r._1)[:70],
                                          str(r._2), pct, uk))
        except Exception:
            continue
    return out


def amostras_noticias(n_max, seed):
    rng = random.Random(seed)
    out = []
    # artigos de navegação (parquet)
    f = os.path.join(DADOS, "noticias", "shipping_news_articles.parquet")
    try:
        df = pd.read_parquet(f, columns=["date", "title", "content"])
        df = df[["date", "title", "content"]]              # ordem explícita
        idx = list(range(len(df)))
        rng.shuffle(idx)
        for i in idx[: n_max // 2]:
            d, t, c = df.iloc[i]
            texto = " ".join(str(c).split()[:220])     # ~primeiros 220 termos
            out.append(f"NOTICIA navegacao {str(d)[:10]} | {t} | {texto}")
    except Exception as e:
        print(f"  sem shipping_news: {e}")
    # notícias de petróleo (csv)
    f = os.path.join(DADOS, "noticias", "crude_oil_news.csv")
    try:
        csv.field_size_limit(10_000_000)
        df = pd.read_csv(f, usecols=["pubDate", "title", "content"])
        df = df[["pubDate", "title", "content"]]           # ordem explícita
        idx = list(range(len(df)))
        rng.shuffle(idx)
        for i in idx[: n_max // 2]:
            d, t, c = df.iloc[i]
            texto = " ".join(str(c).split()[:220])
            out.append(f"NOTICIA petroleo {str(d)[:10]} | {t} | {texto}")
    except Exception as e:
        print(f"  sem crude_oil_news: {e}")
    return out

# ---------------------------------------------------------------- main

def main():
    global DADOS, COMEX, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frete", type=int, default=400_000)
    ap.add_argument("--n-unctad", type=int, default=60_000)
    ap.add_argument("--n-noticias", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dados", default=DADOS, help="dir das bases (default: ./dados)")
    ap.add_argument("--out", default=OUT, help="dir de saída (default: ./treino/corpus)")
    args = ap.parse_args()

    DADOS = args.dados
    COMEX = os.path.join(DADOS, "comex")
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    print("[1/4] decodificadores...")
    ncm_dic, pais_dic = carregar_decodificadores()
    brent_m = carregar_brent_mensal()

    print("[2/4] amostras de frete (Comex IMP marítimo)...")
    frete = amostras_comex(ncm_dic, pais_dic, brent_m, args.n_frete, args.seed)
    print(f"  → {len(frete)} amostras")

    print("[3/4] amostras UNCTAD exportação + notícias...")
    unctad = amostras_unctad(args.n_unctad, args.seed)
    notic = amostras_noticias(args.n_noticias, args.seed)
    print(f"  → {len(unctad)} exportação, {len(notic)} notícias")

    tudo = frete + unctad + notic
    rng = random.Random(args.seed)
    rng.shuffle(tudo)
    n_val = max(1_000, int(0.01 * len(tudo)))
    val, train = tudo[:n_val], tudo[n_val:]
    with open(os.path.join(OUT, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train))
    with open(os.path.join(OUT, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val))
    print(f"[4/4] corpus salvo: {len(train)} treino / {len(val)} validação em {OUT}")

    # tokenizer BPE treinado do zero
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=16_000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(tudo, trainer=trainer, length=len(tudo))
    tok.save(os.path.join(OUT, "tokenizer.json"))
    print(f"tokenizer salvo (vocab={tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
