# -*- coding: utf-8 -*-
# ============================================================================
# TREINAR O GPT DE FRETES DO ZERO, NA GPU DO KAGGLE — kaggle_treinar_gpt.py
# ============================================================================
# Instala o pacote navi direto do repo (github.com/danttis/ldm) e treina o
# mesmo GPT decoder-only do treino local — mas em GPU T4/P100 (16 GB), onde
# cabe uma configuração ~3× maior e o treino leva ~1 h em vez de ~7 h em CPU.
#
# Passo a passo:
#   1. Suba treino/fretes-corpus.zip como dataset Kaggle chamado "fretes-corpus"
#      (o mesmo usado pelos outros notebooks — se já subiu, pule)
#   2. Notebook novo → Add Input → fretes-corpus → Settings → GPU (T4 ×2 ou P100)
#   3. Cole as células abaixo e rode Run All
#   4. No fim: o modelo (.pt) fica em /kaggle/working/checkpoints —
#      baixe pela aba Output e use localmente com:
#        navi-gerar --ckpts <pasta baixada> --tokenizer treino/corpus/tokenizer.json "FRETE ... =>"
#
# Config GPU: emb 384, 8 camadas, bloco 512, lote 48 ≈ 21 M parâmetros
# (a local é emb 256, 6 camadas, bloco 128, lote 16 ≈ 9 M — limitada pela RAM)
# ============================================================================

# %% [1] Instala o pacote navi (torch já vem no Kaggle) --------------------------
!pip -q install "git+https://github.com/danttis/ldm.git"

# %% [2] Prepara o corpus: descompacta o zip do dataset + baixa o tokenizer ------
import glob, os, urllib.request, zipfile

os.makedirs("/kaggle/working/corpus", exist_ok=True)

# a) o zip do dataset (se subiu o zip) ou os arquivos direto (se descompactou)
zp = glob.glob("/kaggle/input/**/fretes-corpus.zip", recursive=True)
if zp:
    with zipfile.ZipFile(zp[0]) as z:
        z.extractall("/kaggle/working/corpus")
    print("extraído:", zp[0])
tr = glob.glob("/kaggle/input/**/train.txt", recursive=True)
if tr and not os.path.exists("/kaggle/working/corpus/train.txt"):
    src = os.path.dirname(tr[0])              # dataset já descompactado
    for fn in ["train.txt", "val.txt"]:
        destino = f"/kaggle/working/corpus/{fn}"
        if os.path.exists(os.path.join(src, fn)) and not os.path.exists(destino):
            os.symlink(os.path.join(src, fn), destino)
    print("dataset já descompactado:", src)

# b) tokenizer.json (fica no repo, não no zip)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/danttis/ldm/danttis/treino/corpus/tokenizer.json",
    "/kaggle/working/corpus/tokenizer.json")
print("arquivos:", sorted(os.listdir("/kaggle/working/corpus")))

# %% [3] Treina na GPU ------------------------------------------------------------
# ~21 M parâmetros; T4/P100: cerca de 4–8 passos/s → 20 mil passos ≈ 40–80 min.
# Checkpoints a cada 500 passos em /kaggle/working/checkpoints (resume ok).
!navi-treinar --corpus /kaggle/working/corpus --ckpts /kaggle/working/checkpoints \
    --emb 384 --camadas 8 --bloco 512 --lote 48 --passos 20000

# %% [4] Testa uma estimação com o modelo treinado ---------------------------------
!navi-gerar --ckpts /kaggle/working/checkpoints \
    --tokenizer /kaggle/working/corpus/tokenizer.json \
    --temp 0.3 \
    "FRETE maritimo importacao | ano=2025 | mes=03 | produto=OLEO BRUTO DE PETROLEO NCM=27090010 | origem=Arabia Saudita | uf=RS | peso_kg=500_000 | fob_usd=250_000 | brent=85 =>"

# %% [5] Libera espaço: apaga os checkpoints intermediários, só o final fica ----
import glob as g, os
cks = sorted(g.glob("/kaggle/working/checkpoints/modelo_*.pt"),
             key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
for f in cks[:-1]:
    os.remove(f)
print("mantido:", cks[-1], f"({os.path.getsize(cks[-1])/1e6:.0f} MB)")
print("baixe o modelo pela aba Output — o intervalo de confiança sai com --temp maior")

# %% [6] (opcional) Continuar o treino por mais passos depois ----------------------
# !navi-treinar --corpus /kaggle/working/corpus --ckpts /kaggle/working/checkpoints \
#     --emb 384 --camadas 8 --bloco 512 --lote 48 --passos 40000 --resume
