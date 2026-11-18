#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
treinar_llm.py — Treina, do zero, um modelo de linguagem estilo GPT para fretes marítimos.

Arquitetura: Transformer decoder-only (causal) implementado em PyTorch puro.
  - embeddings de token + posição (aprendidos)
  - N blocos: attention (multi-head) + MLP, pré-normalização (LayerNorm), residual
  - saída: logits sobre o vocabulário (16k, BPE treinado em montar_dataset.py)

Como usar (após montar_dataset.py):
  python treinar_llm.py --steps 2000          # treino padrão (CPU: horas; GPU: minutos)
  python treinar_llm.py --resume              # continua do último checkpoint
  python gerar.py "FRETE maritimo importacao | ano=2025 | mes=03 | produto=fertilizantes NCM=310210 | origem=Russia | uf=RS | peso_kg=500_000 | fob_usd=150_000 | brent=85 =>"
"""
import argparse
import glob
import math
import os
import time
from array import array

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(RAIZ, "treino", "corpus")
CKPT_DIR = os.path.join(RAIZ, "treino", "checkpoints")

# ------------------------------------------------------------------ modelo

class Bloco(nn.Module):
    def __init__(self, emb, cab, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb)
        self.attn = nn.MultiheadAttention(emb, cab, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(emb)
        self.mlp = nn.Sequential(
            nn.Linear(emb, 4 * emb), nn.GELU(), nn.Dropout(drop),
            nn.Linear(4 * emb, emb), nn.Dropout(drop))

    def forward(self, x, mascara):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mascara, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab, cfg["emb"])
        self.pos = nn.Embedding(cfg["bloco"], cfg["emb"])
        self.drop = nn.Dropout(cfg["drop"])
        self.blocos = nn.ModuleList(
            [Bloco(cfg["emb"], cfg["cab"], cfg["drop"]) for _ in range(cfg["camadas"])])
        self.ln = nn.LayerNorm(cfg["emb"])
        self.cabeca = nn.Linear(cfg["emb"], vocab, bias=False)
        self.cabeca.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mascara = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for b in self.blocos:
            x = b(x, mascara)
        x = self.ln(x)
        logitos = self.cabeca(x)
        perda = None
        if alvos is not None:
            perda = F.cross_entropy(
                logitos.view(-1, logitos.size(-1)), alvos.view(-1), ignore_index=0)
        return logitos, perda

    @torch.no_grad()
    def gerar(self, idx, max_novos=64, temp=0.8, top_k=20):
        for _ in range(max_novos):
            janela = idx[:, -self.cfg["bloco"]:]
            logitos, _ = self(janela)
            logitos = logitos[:, -1] / temp
            if top_k:
                v, _ = torch.topk(logitos, min(top_k, logitos.size(-1)))
                logitos[logitos < v[:, [-1]]] = -float("inf")
            prob = F.softmax(logitos, dim=-1)
            idx = torch.cat([idx, torch.multinomial(prob, 1)], dim=1)
        return idx

# ------------------------------------------------------------------ dados

class CorpusFretes(Dataset):
    """Tokeniza o corpus 1× na RAM e entrega blocos aleatórios de tokens."""

    def __init__(self, arq, tok, bloco, limite_amostras=None):
        # array('i'): 4 bytes/token. Uma lista Python de ~40M ints consome
        # ~1,4 GB (cada int é um objeto) e estoura máquinas com 8 GB de RAM.
        ids = array("i")
        pad = tok.token_to_id("<pad>")
        print(f"tokenizando {arq} ...")
        with open(arq, encoding="utf-8") as f:
            linhas = f.readlines()
        if limite_amostras:
            linhas = linhas[:limite_amostras]
        for i, l in enumerate(linhas):
            ids.extend(tok.encode(l.rstrip("\n")).ids)
            ids.append(pad)
            if i % 100_000 == 0:
                print(f"  {i} linhas, {len(ids):,} tokens")
        self.ids = np.array(ids, dtype=np.int32)
        self.bloco = bloco
        print(f"  total: {len(self.ids):,} tokens")

    def __len__(self):
        return max(1, (len(self.ids) - 1) // self.bloco)

    def __getitem__(self, i):
        j = i * self.bloco
        x = self.ids[j : j + self.bloco]
        y = self.ids[j + 1 : j + self.bloco + 1]
        if len(y) < self.bloco:
            y = np.append(y, 0)
        if len(x) < self.bloco:
            x = np.append(x, 0)
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

# ------------------------------------------------------------------ treino

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--cab", type=int, default=8)
    ap.add_argument("--camadas", type=int, default=6)
    ap.add_argument("--bloco", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--passos", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-cada", type=int, default=200)
    ap.add_argument("--amostras", type=int, default=None,
                    help="limita nº de linhas do corpus (treino rápido de teste)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev}")

    tok = Tokenizer.from_file(os.path.join(CORPUS, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    cfg = {"emb": args.emb, "cab": args.cab, "camadas": args.camadas,
           "bloco": args.bloco, "drop": args.drop}

    tr = CorpusFretes(os.path.join(CORPUS, "train.txt"), tok, args.bloco, args.amostras)
    va = CorpusFretes(os.path.join(CORPUS, "val.txt"), tok, args.bloco,
                      args.amostras or 20_000)
    lote_tr = DataLoader(tr, batch_size=args.lote, shuffle=True,
                         num_workers=2, drop_last=True)
    lote_va = DataLoader(va, batch_size=args.lote, shuffle=False, num_workers=1)

    modelo = GPT(vocab, cfg).to(dev)
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros: {n_params:,}")

    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    passo_inicial = 0
    # ordena numericamente (lexicográfico pegaria modelo_80 depois de modelo_20000)
    ultimo = sorted(glob.glob(os.path.join(CKPT_DIR, "modelo_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if args.resume and ultimo:
        st = torch.load(ultimo[-1], map_location=dev)
        modelo.load_state_dict(st["modelo"]); opt.load_state_dict(st["opt"])
        passo_inicial = st["passo"]; cfg.update(st["cfg"])
        print(f"retomando de {os.path.basename(ultimo[-1])} (passo {passo_inicial})")

    # cronograma coseno com aquecimento
    aquec = min(200, args.passos // 10 + 1)
    def lr_esc(p):
        if p < aquec:
            return (p + 1) / aquec
        pro = (p - aquec) / max(1, args.passos - aquec)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * pro))

    esc = torch.optim.lr_scheduler.LambdaLR(opt, lr_esc)

    print("iniciando treino...")
    passo, t0 = passo_inicial, time.time()
    rodando = True
    while rodando:
        for x, y in lote_tr:
            x, y = x.to(dev), y.to(dev)
            _, perda = modelo(x, y)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step(); esc.step()
            passo += 1
            if passo % 50 == 0:
                print(f"passo {passo}/{args.passos} | perda {perda.item():.4f} "
                      f"| lr {opt.param_groups[0]['lr']:.2e} "
                      f"| {passo/(time.time()-t0):.1f} passos/s", flush=True)
            if passo % args.val_cada == 0:
                modelo.eval(); vp = []
                with torch.no_grad():
                    for i, (xv, yv) in enumerate(lote_va):
                        if i >= 20: break
                        _, p = modelo(xv.to(dev), yv.to(dev)); vp.append(p.item())
                print(f"  >> validação: {np.mean(vp):.4f}", flush=True)
                modelo.train()
            if passo % 500 == 0 or passo >= args.passos:
                arq = os.path.join(CKPT_DIR, f"modelo_{passo}.pt")
                torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(),
                            "cfg": cfg, "passo": passo, "vocab": vocab}, arq)
                print(f"  salvo {arq}", flush=True)
            if passo >= args.passos:
                rodando = False; break
    print(f"concluído: {passo} passos em {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
