#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar.py — Usa a LLM de fretes treinada para estimar frete a partir do contexto.

Exemplos:
  python gerar.py "FRETE maritimo importacao | ano=2025 mes=03 | produto=FERTILIZANTES NCM=31021000 | origem=Russia uf=RS | peso_kg=500_000 fob_usd=150_000 brent=85 =>"
  python gerar.py "FRETE maritimo exportacao Brasil | ano=2020 | produto=soja | destino=China | custo_pct_fob=9.5 =>"

O modelo completa o texto — o número após frete_usd_por_kg é a estimativa.
"""
import argparse
import glob
import os

import torch
from tokenizers import Tokenizer

import treinar_llm as T   # reusa a classe GPT

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(RAIZ, "treino", "checkpoints")
CORPUS = os.path.join(RAIZ, "treino", "corpus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--max", type=int, default=30, help="tokens a gerar")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--ckpt", default=None, help="checkpoint específico (default: último)")
    args = ap.parse_args()

    alvo = args.ckpt or sorted(glob.glob(os.path.join(CKPT_DIR, "modelo_*.pt")))[-1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(alvo, map_location=dev)
    modelo = T.GPT(st["vocab"], st["cfg"]).to(dev)
    modelo.load_state_dict(st["modelo"]); modelo.eval()

    tok = Tokenizer.from_file(os.path.join(CORPUS, "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        saida = modelo.gerar(x, max_novos=args.max, temp=args.temp)
    print(tok.decode(saida[0].tolist()))


if __name__ == "__main__":
    main()
