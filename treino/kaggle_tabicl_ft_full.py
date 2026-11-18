# -*- coding: utf-8 -*-
# ============================================================================
# FINE-TUNING DO TABICL v2 COM TODOS OS DADOS — kaggle_tabicl_ft_full.py
# ============================================================================
# Responde: "e se o modelo de fundação usar TUDO (~370 mil linhas), ele alcança
# o LightGBM?" — COM A COMPARAÇÃO LIMPA: mesmo conjunto de treino para os dois,
# mesmo teste de 20 mil (idx[30k:50k], IGUAL ao do run anterior de 30k).
#
# ⚠️ IMPORTANTE — o "LightGBM 300k = 0.337" do run anterior estava CONTAMINADO:
# idx[:300_000] incluía as 20 mil linhas de teste (vazamento). Descarte esse
# número. Este script treina o LightGBM nos MESMOS dados do FT, sem teste dentro.
#
# Experimento (mesmo teste dos runs anteriores):
#   A. TabICL v2 zero-shot ICL      (10 mil de contexto)
#   B. TabICL v2 FINE-TUNED COM TUDO (~370 mil linhas, gradiente)
#   D. LightGBM                     (as mesmas ~370 mil linhas, sem o teste)
#
# ⏱️ Tempo esperado no T4/P100: B roda em chunks de 10k (~37 chunks/época);
#    10 épocas ≈ 8–14 h → USE P100 (sessão de 12 h) ou T4×2 (9 h) e deixe
#    time_limit=6h como trava de segurança (o melhor modelo é salvo antes).
#
# Uso: igual aos outros — dataset c/ train.txt como Input, GPU ligada, Run All.
# ============================================================================

# %% [1] Instalação -----------------------------------------------------------
!pip -q install "tabicl[finetune]" lightgbm

# %% [2] Configuração e carga do corpus ----------------------------------------
import glob, os, re, random
import numpy as np
import pandas as pd
import torch

CAMINHOS = glob.glob("/kaggle/input/**/train.txt", recursive=True)  # qualquer profundidade
assert CAMINHOS, "train.txt não achado — anexe o dataset (fretes-corpus/dados-fvr) como Input"
CORPUS = os.path.dirname(CAMINHOS[0])

N_VAL    = 5_000      # early stopping (fora do teste)
N_ICL    = 10_000     # contexto do zero-shot
N_TESTE  = 20_000
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
for c in CAT:
    df[c] = df[c].astype("category")           # categorias globais (1× no dataset)

rng = random.Random(SEED)                        # MESMA semente/split dos runs anteriores
idx = list(range(len(df))); rng.shuffle(idx)

i_teste = idx[30_000 : 30_000 + N_TESTE]         # MESMO teste do run de 30k (comparação direta)
i_val   = idx[:N_VAL]                            # early stopping
i_fit   = idx[N_VAL : 30_000] + idx[30_000 + N_TESTE :]   # TUDO, menos validação e teste
i_icl   = idx[:N_ICL]

def X_de(base):
    return base[COLS].copy()

X_fit, y_fit = X_de(df.iloc[i_fit]), df.iloc[i_fit]["alvo"].to_numpy()
X_val, y_val = X_de(df.iloc[i_val]), df.iloc[i_val]["alvo"].to_numpy()
X_icl, y_icl = X_de(df.iloc[i_icl]), df.iloc[i_icl]["alvo"].to_numpy()
X_te,  y_te  = X_de(df.iloc[i_teste]), df.iloc[i_teste]["alvo"].to_numpy()
real_usd = np.expm1(y_te)
print(f"fit COM TUDO: {X_fit.shape} | val: {X_val.shape} | teste: {X_te.shape} (mesmo do run anterior)")
print("frete/kg real — mediana %.3f | p90 %.3f" %
      (np.median(real_usd), np.quantile(real_usd, 0.9)))

# %% [4] A. TabICL v2 zero-shot (ICL) — referência do run anterior ----------------
from tabicl import TabICLRegressor

icl = TabICLRegressor(random_state=SEED)
icl.fit(X_icl, y_icl)
pa = np.clip(np.expm1(icl.predict(X_te)), 0, None)
qa = np.asarray(icl.predict(X_te, output_type="quantiles", alphas=[0.1, 0.9]))
print(f"TABICL ICL (10 mil contexto)     | MAE {np.abs(pa - real_usd).mean():.4f} US$/kg "
      f"| cobertura 80%: {((real_usd >= np.expm1(qa[:,0])) & (real_usd <= np.expm1(qa[:,1]))).mean():.1%}")

# %% [5] B. TabICL v2 FINE-TUNED COM TUDO (~370 mil, gradiente) -------------------
from tabicl import FinetunedTabICLRegressor

ft = FinetunedTabICLRegressor(
    epochs=10,                    # ~37 chunks/época: 10 épocas ≈ o mesmo volume de
                                 # gradiente de ~150 épocas do run de 30k — suficiente
    learning_rate=1e-5,
    eval_metric="mae",
    early_stopping=True, patience=3,
    time_limit=6 * 3600,          # trava: salva o melhor modelo e para antes do fim da sessão
    save_interval=1,
    random_state=SEED, device=DEVICE, verbose=True,
)
ft.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)

pb = np.clip(np.expm1(np.asarray(ft.predict(X_te)).ravel()), 0, None)
qb = np.asarray(ft.predict(X_te, output_type="quantiles", alphas=[0.1, 0.9]))
print(f"TABICL FT COM TUDO (~370 mil)    | MAE {np.abs(pb - real_usd).mean():.4f} US$/kg "
      f"| cobertura 80%: {((real_usd >= np.expm1(qb[:,0])) & (real_usd <= np.expm1(qb[:,1]))).mean():.1%}")

import joblib
joblib.dump(ft, "/kaggle/working/tabicl_fretes_ft_tudo.joblib")
print("modelo salvo em /kaggle/working/tabicl_fretes_ft_tudo.joblib")

# %% [6] D. LightGBM — MESMAS linhas do FT (comparação limpa, sem teste) ----------
from lightgbm import LGBMRegressor

lgbm_full = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127,
                          random_state=SEED)
lgbm_full.fit(X_fit, y_fit)
pd_full = np.clip(np.expm1(lgbm_full.predict(X_te)), 0, None)
print(f"LIGHTGBM (mesmas ~370 mil)       | MAE {np.abs(pd_full - real_usd).mean():.4f} US$/kg")

# %% [7] Resumo -------------------------------------------------------------------
print("\n=== RESUMO (MAE em US$/kg, MESMO teste do run anterior) ===")
resumo = pd.DataFrame({
    "modelo": ["TabICL ICL 10k", "TabICL FT TUDO", "LightGBM TUDO (limpo)"],
    "mae": [np.abs(p - real_usd).mean() for p in [pa, pb, pd_full]],
    "rmse": [np.sqrt(((p - real_usd) ** 2).mean()) for p in [pa, pb, pd_full]],
})
print(resumo.round(4).to_string(index=False))
print("""
Referências do run anterior (MESMO teste):
  TabICL ICL 10k   = 0.553  ✓ comparável
  TabICL FT 30k    = 0.539  (a pergunta deste run: quanto "tudo" melhora isso?)
  LightGBM 300k    = 0.337  ✗ DESCARTE — treinou com as linhas de teste dentro (vazamento)

Se o FT TUDO encostar no LightGBM TUDO, o modelo de fundação vira o melhor dos dois
(quantis de graça); se perder, LightGBM continua campeão de escala.""")

# %% [8] Estimar um frete novo ------------------------------------------------------
nova = pd.DataFrame([{
    "ncm": "27090010",            # óleo bruto de petróleo
    "origem": "Arábia Saudita",
    "uf": "RJ",
    "ano": 2026.0, "mes": 3.0,
    "peso_kg": 250_000_000.0,     # 250 mil t
    "fob_usd": 130_000_000.0,
    "brent": 85.0,
}])
nova_tfm = nova[COLS].copy()
for c in CAT:
    nova_tfm[c] = pd.Categorical(nova[c], categories=df[c].cat.categories
                                 ).codes.astype(int)   # pd.Categorical: .codes direto
for c in NUM:
    nova_tfm[c] = nova_tfm[c].astype(float)

est = float(np.expm1(np.asarray(ft.predict(nova_tfm)).ravel()[0]))
q = np.asarray(ft.predict(nova_tfm, output_type="quantiles", alphas=[0.1, 0.9])).ravel()
print("frete estimado: %.4f US$/kg (US$ %.1f/t)" % (est, est * 1000))
print("intervalo 80%%: [%.4f, %.4f] US$/kg" % (np.expm1(q[0]), np.expm1(q[1])))
