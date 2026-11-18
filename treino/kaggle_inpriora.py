# -*- coding: utf-8 -*-
# ============================================================================
# FINE-TUNING (IN-CONTEXT) DO INPRIORA NOS DADOS DE FRETE — PRONTO PARA KAGGLE
# ============================================================================
# Inpriora (https://github.com/ai-for-data/inpriora) é um modelo de fundação
# TABULAR: não existe fine-tuning por gradiente. A adaptação ao domínio é
# "in-context": o .fit(X, y) fornece seus dados de frete como contexto e o
# modelo generaliza para novas consultas. Este script faz isso com o corpus
# de fretes marítimos que já montamos (fretes reais declarados à Receita).
#
# Como usar no Kaggle:
#   1. Suba treino/fretes-corpus.zip como dataset Kaggle ("fretes-corpus")
#   2. Notebook novo → anexe o dataset como Input → GPU (T4/P100) ou CPU
#   3. Cole as células e rode Run All
#
# Tarefa: regressão do frete marítimo (US$/kg) a partir de
#   ano, mês, produto (NCM), país de origem, UF, peso, FOB e preço do Brent.
# ============================================================================

# %% [1] Instalação -----------------------------------------------------------
!pip -q install "git+https://github.com/ai-for-data/inpriora.git" lightgbm

# %% [2] Configuração e carga do corpus ----------------------------------------
import glob, os, re, random
import numpy as np
import pandas as pd

CAMINHOS = glob.glob("/kaggle/input/**/train.txt", recursive=True)  # qualquer profundidade
assert CAMINHOS, "Anexe o dataset fretes-corpus (train.txt/val.txt)!"
CORPUS = os.path.dirname(CAMINHOS[0])

N_CONTEXTO  = 10_000     # nº de linhas dadas como contexto ao Inpriora
                            # (modelo ICL: contexto maior = melhor, porém mais lento)
N_TESTE     = 20_000     # amostras de teste
SEED        = 42

def parse_linha(l):
    """'FRETE ... | campo=valor ... => frete_usd=X | frete_usd_por_kg=Y' → dict."""
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
# produto: usa o NCM (8 dígitos) como categoria — mais estável que o nome
df["ncm"] = df["produto"].str.extract(r"NCM=(\d+)").fillna("0")
for c in ["peso_kg", "fob_usd"]:
    df[c] = pd.to_numeric(df[c].str.replace("_", "", regex=False), errors="coerce")
df["brent"] = pd.to_numeric(df["brent"], errors="coerce")
for c in ["ano", "mes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["frete_kg"] = pd.to_numeric(df["frete_usd_por_kg"], errors="coerce")
df["frete_usd"] = pd.to_numeric(
    df["frete_usd"].str.replace("_", "", regex=False), errors="coerce")

df = df.dropna(subset=["frete_kg", "peso_kg", "fob_usd", "ncm", "origem", "uf",
                       "ano", "mes"])
# frete/kg é fortemente assimétrico (mediana ~0,22, p90 ~1,0) → treinar no log
df["alvo"] = np.log1p(df["frete_kg"])

CAT = ["ncm", "origem", "uf"]
NUM = ["ano", "mes", "peso_kg", "fob_usd", "brent"]

def matrizes(base):
    """X_icl: strings (Inpriora codifica categorias nativamente).
       X_lgbm: dtypes 'category' (requisito do LightGBM)."""
    X_icl = base[CAT + NUM].copy()
    for c in NUM:
        X_icl[c] = X_icl[c].astype(float)
    X_lgbm = X_icl.copy()
    for c in CAT:
        X_lgbm[c] = X_lgbm[c].astype("category")
    return X_icl, X_lgbm, base["alvo"].to_numpy()

rng = random.Random(SEED)
idx = list(range(len(df))); rng.shuffle(idx)
i_ctx   = idx[:N_CONTEXTO]            # contexto do Inpriora (= "fine-tuning")
i_teste = idx[N_CONTEXTO : N_CONTEXTO + N_TESTE]
Xc_icl, Xc_lgbm, y_ctx = matrizes(df.iloc[i_ctx])
Xt_icl, Xt_lgbm, y_te  = matrizes(df.iloc[i_teste])
print("contexto:", Xc_icl.shape, "| teste:", Xt_icl.shape)
print("frete/kg real — mediana %.3f | p90 %.3f" %
      (np.expm1(y_te).median(), np.expm1(y_te).quantile(0.9)))

# %% [4] "Fine-tuning" do Inpriora (in-context) + predição -----------------------
import time
from inpriora import InprioraRegressor

modelo = InprioraRegressor(
    n_estimators=8,          # ensemble (padrão); suba p/ 16 se tiver tempo
    random_seed=SEED,
    verbose=True,
)
t0 = time.time()
modelo.fit(Xc_icl, y_ctx)   # adaptação ao domínio: dados de frete como contexto
print(f"fit (contexto) em {time.time()-t0:.0f}s")

t0 = time.time()
# média + quantis (incerteza de graça — distribuição de buckets)
media = np.asarray(modelo.predict(Xt_icl))                    # default = média
quantis = np.asarray(modelo.predict(Xt_icl, output_type="quantiles",
                                    alphas=[0.1, 0.9]))       # (n, 2)
q10, q90 = quantis[:, 0], quantis[:, 1]
print(f"predição em {time.time()-t0:.0f}s")

real_usd = np.expm1(y_te)
pred_usd = np.clip(np.expm1(media), 0, None)

mae  = np.abs(pred_usd - real_usd).mean()
rmse = np.sqrt(((pred_usd - real_usd) ** 2).mean())
dentro = ((real_usd >= np.expm1(q10)) & (real_usd <= np.expm1(q90))).mean()
print(f"\nINPRIORA  | MAE {mae:.4f} US$/kg | RMSE {rmse:.4f} | "
      f"cobertura 80%: {dentro:.1%}")

# %% [5] Baseline LightGBM (mesmo contexto — comparação justa) -------------------
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127,
                     random_state=SEED)
lgbm.fit(Xc_lgbm, y_ctx)
pl = np.clip(np.expm1(lgbm.predict(Xt_lgbm)), 0, None)
print(f"LIGHTGBM (mesmo contexto de {N_CONTEXTO:,}) | "
      f"MAE {np.abs(pl - real_usd).mean():.4f} US$/kg")

# LightGBM com TODAS as linhas (ICL não escala igual — modelo cresce junto)
# 300 mil SEM as linhas de teste (idx[:300_000] continha o teste idx[N_CONTEXTO:N_CONTEXTO+N_TESTE]
# e vazava no treino — corrige o baseline "LightGBM 300 mil")
i_resto = idx[:N_CONTEXTO] + idx[N_CONTEXTO + N_TESTE : 300_000 + N_TESTE]
Xr_icl, Xr_lgbm, yr = matrizes(df.iloc[i_resto])
lgbm_full = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127,
                          random_state=SEED)
lgbm_full.fit(Xr_lgbm, yr)
pf = np.clip(np.expm1(lgbm_full.predict(Xt_lgbm)), 0, None)
print(f"LIGHTGBM (300 mil linhas)         | "
      f"MAE {np.abs(pf - real_usd).mean():.4f} US$/kg")

# %% [6] Análise do erro por grupo (onde o modelo acerta/erra) -------------------
res = pd.DataFrame({
    "origem": df.iloc[i_teste]["origem"].to_numpy(),
    "uf": df.iloc[i_teste]["uf"].to_numpy(),
    "ano": df.iloc[i_teste]["ano"].to_numpy(),
    "real": real_usd, "inpriora": pred_usd, "lgbm": pf,
})
res["erro_inpriora"] = (res["inpriora"] - res["real"]).abs()
res["erro_lgbm"] = (res["lgbm"] - res["real"]).abs()
print("\nMAE (US$/kg) por país de origem — top 8 volume:")
vol = res.groupby("origem").agg(n=("real", "size"),
                                inpriora=("erro_inpriora", "mean"),
                                lgbm=("erro_lgbm", "mean"))
print(vol.sort_values("n", ascending=False).head(8).round(4).to_string())

print("\nMAE por ano:")
print(res.groupby("ano")[["erro_inpriora", "erro_lgbm"]].mean()
      .round(4).to_string())

# %% [7] Estimar um frete novo (uso do modelo) ----------------------------------
nova = pd.DataFrame([{
    "ncm": "27090010",            # óleo bruto de petróleo
    "origem": "Arábia Saudita",
    "uf": "RJ",
    "ano": 2026.0, "mes": 3.0,
    "peso_kg": 250_000_000.0,     # 250 mil t
    "fob_usd": 130_000_000.0,
    "brent": 85.0,
}])[Xc_icl.columns]               # mesma ordem de colunas do contexto
media_nova = float(np.expm1(np.asarray(modelo.predict(nova)).ravel()[0]))
print("frete estimado: %.4f US$/kg (US$ %.1f/t)" % (media_nova, media_nova * 1000))
q = np.asarray(modelo.predict(nova, output_type="quantiles",
                              alphas=[0.1, 0.9])).ravel()
print("intervalo 80%%: [%.4f, %.4f] US$/kg" % (np.expm1(q[0]), np.expm1(q[1])))
