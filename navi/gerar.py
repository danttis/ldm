#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gerar.py — Usa a LLM de fretes treinada para estimar frete a partir do contexto.

Exemplos:
  navi-gerar "FRETE maritimo importacao | ano=2025 | mes=03 | produto=FERTILIZANTES NCM=31021000 | origem=Russia uf=RS | peso_kg=500_000 fob_usd=150_000 brent=85 =>"
  navi-gerar "FRETE maritimo exportacao Brasil | ano=2020 | produto=soja | destino=China | custo_pct_fob=9.5 =>"

O modelo completa o texto — o número após frete_usd_por_kg é a estimativa.
"""
import argparse
import glob
import os

import torch
from tokenizers import Tokenizer, decoders

from navi import treinar_llm as T   # reusa a classe GPT

CORPUS = os.path.join("treino", "corpus")         # padrões relativos ao CWD
CKPT_DIR = os.path.join("treino", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--max", type=int, default=30, help="tokens a gerar")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--ckpt", default=None, help="checkpoint específico (default: último)")
    ap.add_argument("--ckpts", default=CKPT_DIR, help="dir de checkpoints")
    ap.add_argument("--tokenizer", default=os.path.join(CORPUS, "tokenizer.json"))
    args = ap.parse_args()

    # ordena numericamente (lexicográfico pegaria modelo_80 depois de modelo_20000)
    ckpts = sorted(glob.glob(os.path.join(args.ckpts, "modelo_*.pt")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    alvo = args.ckpt or (ckpts[-1] if ckpts else None)
    if alvo is None:
        raise SystemExit(f"nenhum checkpoint em {args.ckpts} — treine primeiro "
                         f"(navi-treinar) ou passe --ckpt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(alvo, map_location=dev)
    modelo = T.GPT(st["vocab"], st["cfg"]).to(dev)
    modelo.load_state_dict(st["modelo"]); modelo.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    tok.decoder = decoders.ByteLevel()   # sem isso o texto sai com "Ġ" no lugar dos espaços
    ids = tok.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        saida = modelo.gerar(x, max_novos=args.max, temp=args.temp)
    print(tok.decode(saida[0].tolist()))


if __name__ == "__main__":
    main()
