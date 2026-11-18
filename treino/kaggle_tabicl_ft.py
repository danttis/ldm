# -*- coding: utf-8 -*-
# ============================================================================
# FINE-TUNING (GRADIENTE) DO TABICL v2 NOS DADOS DE FRETE — KAGGLE
# ============================================================================
# TabICL v2 (SODA/INRIA — open source, 28M params, sem Prior Labs) tem
# fine-tuning OFICIAL com gradiente: FinetunedTabICLRegressor (perda pinball
# nos 999 quantis, AdamW + early stopping). E prevê quantis — intervalo de
# confiança de graça, igual ao Inpriora.
#
# Experimento — 4 regimes, mesmo conjunto de teste (20 mil fretes reais):
#   A. TabICL v2 zero-shot ICL   (10 mil linhas de contexto, pesos congelados)
#   B. TabICL v2 FINE-TUNED      (30 mil linhas, gradiente — o foco)
#   C. LightGBM                  (mesmas 30 mil linhas — comparação justa)
#   D. LightGBM                  (300 mil linhas — referência de escala)
#
# Como usar no Kaggle:
#   1. Suba treino/fretes-corpus.zip como dataset Kaggle ("fretes-corpus")
#   2. Notebook novo → anexe o dataset como Input → GPU T4/P100
#   3. Cole as células e rode Run All
# ============================================================================

# %% [1] Instalação -----------------------------------------------------------
!pip -q install "tabicl[finetune]" lightgbm

# %% [2] Configuração e carga do corpus ----------------------------------------
import glob, os, re, random
import numpy as np
import pandas as pd
import torch

CAMINHOS = glob.glob("/kaggle/input/*/train.txt")
assert CAMINHOS, "Anexe o dataset fretes-corpus (train.txt/val.txt)!"
CORPUS = os.path.dirname(CAMINHOS[0])

N_FT     = 30_000     # linhas para o FINE-TUNING
N_VAL    = 5_000      # retiradas do conjunto de FT p/ early stopping (não do teste!)
N_ICL    = 10_000     # linhas de contexto do zero-shot
N_TESTE  = 20_000     # amostras de teste (iguais em todos os regimes)
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
print("dispositivo:", DEVICE)

def parse_linha(l):
    d = {}
    contexto, _, alvo = l.partition("=>")
    for parte in contexto.split("|"):
        if "=" in parte:
            k, v = parte.split("=", 1)
            d[k.strip()] = v.strip()
    d.update(dict(re.findall(r"(\w+)=(\S+)", alvo)))
    return d

with open(os.path.join(CORPUS, "train.txt"), encoding="utf-8") as f:
    linhas = [l.rstrip("\n") for l in f if l.startswith("FRETE maritimo importacao")]
print(f"{len(linhas):,} linhas de frete no corpus")

df = pd.DataFrame([parse_linha(l) for l in linhas])

# %% [3] Features e alvo -------------------------------------------------------
# mesmas features do kaggle_inpriora.py: ncm/origem/uf (cat) + ano/mes/peso/fob/brent
df["ncm"] = df["produto"].str.extract(r"NCM=(\d+)").fillna("0")
for c in ["peso_kg", "fob_usd"]:
    df[c] = pd.to_numeric(df[c].str.replace("_", "", regex=False), errors="coerce")
df["brent"] = pd.to_numeric(df["brent"], errors="coerce")
for c in ["ano", "mes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["frete_kg"] = pd.to_numeric(df["frete_usd_por_kg"], errors="coerce")

df = df.dropna(subset=["frete_kg", "peso_kg", "fob_usd", "ncm", "origem", "uf",
                       "ano", "mes"])
df["alvo"] = np.log1p(df["frete_kg"])          # assimétrico → log

CAT = ["ncm", "origem", "uf"]
NUM = ["ano", "mes", "peso_kg", "fob_usd", "brent"]
COLS = CAT + NUM

# categorias com dtype category definidas UMA VEZ no dataset inteiro — o TabICL
# e o LightGBM veem as mesmas categorias em treino/teste/predição
for c in CAT:
    df[c] = df[c].astype("category")

rng = random.Random(SEED)                        # mesma semente/split do inpriora
idx = list(range(len(df))); rng.shuffle(idx)
i_ft    = idx[:N_FT]                             # fine-tune (30k; 5k viram validação)
i_icl   = idx[:N_ICL]                            # contexto zero-shot (subconjunto)
i_val   = i_ft[-N_VAL:]                          # early stopping — fora do teste
i_fit   = i_ft[:-N_VAL]
i_teste = idx[N_FT : N_FT + N_TESTE]             # teste — NUNCA visto

def X_de(base):
    return base[COLS].copy()                     # dtypes: category + float

X_fit, y_fit = X_de(df.iloc[i_fit]), df.iloc[i_fit]["alvo"].to_numpy()
X_val, y_val = X_de(df.iloc[i_val]), df.iloc[i_val]["alvo"].to_numpy()
X_icl, y_icl = X_de(df.iloc[i_icl]), df.iloc[i_icl]["alvo"].to_numpy()
X_te,  y_te  = X_de(df.iloc[i_teste]), df.iloc[i_teste]["alvo"].to_numpy()
real_usd = np.expm1(y_te)
print("fit:", X_fit.shape, "| val:", X_val.shape, "| contexto ICL:", X_icl.shape,
      "| teste:", X_te.shape)
print("frete/kg real — mediana %.3f | p90 %.3f" %
      (np.median(real_usd), np.quantile(real_usd, 0.9)))

# %% [4] A. TabICL v2 zero-shot (ICL, pesos congelados) --------------------------
from tabicl import TabICLRegressor

icl = TabICLRegressor(random_state=SEED)
icl.fit(X_icl, y_icl)                            # fit = armazena contexto (não treina)
pa = np.clip(np.expm1(icl.predict(X_te)), 0, None)
qa = np.asarray(icl.predict(X_te, output_type="quantiles", alphas=[0.1, 0.9]))
print(f"TABICL ICL (10 mil contexto)     | MAE {np.abs(pa - real_usd).mean():.4f} US$/kg "
      f"| cobertura 80%: {((real_usd >= np.expm1(qa[:,0])) & (real_usd <= np.expm1(qa[:,1]))).mean():.1%}")

# %% [5] B. TabICL v2 FINE-TUNED (gradiente) — o foco -----------------------------
from tabicl import FinetunedTabICLRegressor

ft = FinetunedTabICLRegressor(
    epochs=30,                    # passes de meta-batches (chunks de 10k linhas)
    learning_rate=1e-5,           # padrão do pacote (AdamW + coseno c/ warmup)
    eval_metric="mae",            # early stopping no MAE (espaço log)
    early_stopping=True, patience=8,
    random_state=SEED, device=DEVICE, verbose=True,
)
ft.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)   # AQUI o gradiente flui

pb = np.clip(np.expm1(np.asarray(ft.predict(X_te)).ravel()), 0, None)
qb = np.asarray(ft.predict(X_te, output_type="quantiles", alphas=[0.1, 0.9]))
print(f"TABICL FINE-TUNED (30 mil)       | MAE {np.abs(pb - real_usd).mean():.4f} US$/kg "
      f"| cobertura 80%: {((real_usd >= np.expm1(qb[:,0])) & (real_usd <= np.expm1(qb[:,1]))).mean():.1%}")

import joblib
joblib.dump(ft, "/kaggle/working/tabicl_fretes_finetuned.joblib")
print("modelo salvo em /kaggle/working/tabicl_fretes_finetuned.joblib")

# %% [6] C/D. LightGBM — mesma base (justo) e 300 mil (escala) -------------------
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127,
                     random_state=SEED)
lgbm.fit(X_de(df.iloc[i_fit]), y_fit)
pc = np.clip(np.expm1(lgbm.predict(X_te)), 0, None)
print(f"LIGHTGBM (mesmas 30 mil)         | MAE {np.abs(pc - real_usd).mean():.4f} US$/kg")

i_resto = idx[:300_000]
lgbm_full = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127,
                          random_state=SEED)
lgbm_full.fit(X_de(df.iloc[i_resto]), df.iloc[i_resto]["alvo"].to_numpy())
pd_full = np.clip(np.expm1(lgbm_full.predict(X_te)), 0, None)
print(f"LIGHTGBM (300 mil linhas)       | MAE {np.abs(pd_full - real_usd).mean():.4f} US$/kg")

# %% [7] Resumo + análise do erro ------------------------------------------------
print("\n=== RESUMO (MAE em US$/kg, 20 mil fretes de teste) ===")
resumo = pd.DataFrame({
    "modelo": ["TabICL ICL 10k", "TabICL FT 30k", "LightGBM 30k", "LightGBM 300k"],
    "mae": [np.abs(p - real_usd).mean() for p in [pa, pb, pc, pd_full]],
    "rmse": [np.sqrt(((p - real_usd) ** 2).mean()) for p in [pa, pb, pc, pd_full]],
})
print(resumo.round(4).to_string(index=False))
print("\nB vencer A mede o ganho real do gradiente;")
print("encostar em D mostra o custo de escala do modelo fundação.")

res = pd.DataFrame({
    "origem": df.iloc[i_teste]["origem"].to_numpy(),
    "uf": df.iloc[i_teste]["uf"].to_numpy(),
    "ano": df.iloc[i_teste]["ano"].to_numpy(),
    "real": real_usd, "icl": pa, "ft": pb, "lgbm30": pc, "lgbm300": pd_full,
})
for m in ["icl", "ft", "lgbm30", "lgbm300"]:
    res[f"erro_{m}"] = (res[m] - res["real"]).abs()
print("\nMAE por país de origem — top 8 volume:")
vol = res.groupby("origem").agg(
    n=("real", "size"),
    **{f"erro_{m}": (f"erro_{m}", "mean") for m in ["icl", "ft", "lgbm30", "lgbm300"]})
print(vol.sort_values("n", ascending=False).head(8).round(4).to_string())

print("\nMAE por ano (desvio temporal):")
print(res.groupby("ano")[[f"erro_{m}" for m in ["icl", "ft", "lgbm30", "lgbm300"]]]
      .mean().round(4).to_string())

# %% [8] Estimar um frete novo (uso do modelo) -----------------------------------
nova = pd.DataFrame([{
    "ncm": "27090010",            # óleo bruto de petróleo
    "origem": "Arábia Saudita",
    "uf": "RJ",
    "ano": 2026.0, "mes": 3.0,
    "peso_kg": 250_000_000.0,     # 250 mil t
    "fob_usd": 130_000_000.0,
    "brent": 85.0,
}])
for c in CAT:                      # mesmas categorias do treino
    nova[c] = pd.Categorical(nova[c], categories=df[c].cat.categories)
for c in NUM:
    nova[c] = nova[c].astype(float)

est = float(np.expm1(np.asarray(ft.predict(nova)).ravel()[0]))
q = np.asarray(ft.predict(nova, output_type="quantiles", alphas=[0.1, 0.9])).ravel()
print("frete estimado: %.4f US$/kg (US$ %.1f/t)" % (est, est * 1000))
print("intervalo 80%%: [%.4f, %.4f] US$/kg" % (np.expm1(q[0]), np.expm1(q[1])))
