# -*- coding: utf-8 -*-
# ============================================================================
# FINE-TUNING DE LLM ABERTA PARA FRETES MARÍTIMOS — PRONTO PARA KAGGLE
# ============================================================================
# Como usar no Kaggle:
#   1. Suba o corpus como dataset Kaggle:
#        zip treino/corpus/train.txt treino/corpus/val.txt  →  fretes-corpus.zip
#        kaggle datasets create -p pasta/  (ou pela UI: New Dataset → upload)
#   2. Crie um Notebook, anexe o dataset "fretes-corpus" como Input
#      e ative a GPU (T4 x2 ou P100) em Settings → Accelerator
#   3. Cole as células abaixo e rode Run All
#
# Modelo base: Qwen2.5-1.5B (Apache-2.0, ungated, multilíngue — cai bem em pt)
#   → para mais qualidade troque MODELO abaixo por "Qwen/Qwen2.5-7B" (QLoRA cabe
#     em 16 GB); para mais velocidade, "Qwen/Qwen2.5-0.5B".
#   → alternativas: "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.2-3B"
#     (esta exige aceitar licença + token HF).
#
# Nota sobre TabArena (huggingface.co/spaces/TabArena/leaderboard): aquele
# ranking é de modelos TABULARES (TabPFN etc.), sem fine-tuning de linguagem.
# A célula [7] traz um baseline tabular (LightGBM) para comparação — em
# previsão numérica pura ele costuma ganhar da LLM; a LLM ganha em contexto
# livre (notícias, rotas novas, texto).
# ============================================================================

# %% [1] Dependências e configuração ----------------------------------------
!pip -q install -U transformers datasets peft bitsandbytes accelerate sentencepiece

import os, re, glob, random, math
import numpy as np
import torch

MODELO    = "Qwen/Qwen2.5-1.5B"          # base aberta (troque à vontade)
CAMINHO   = glob.glob("/kaggle/input/*/train.txt")          # dataset anexado
assert CAMINHO, "Anexe o dataset fretes-corpus (train.txt/val.txt)!"
CORPUS   = os.path.dirname(CAMINHO[0])
SEq      = 1024                            # tokens por bloco
N_AMOSTRAS = 200_000                       # subamostra p/ caber na sessão Kaggle
EPOCAS   = 1
LORA_R   = 16
LORA_ALFA= 32
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if DEV == "cuda" else "nenhuma")

# %% [2] Carrega o corpus -----------------------------------------------------
def carregar(nome, n):
    with open(os.path.join(CORPUS, nome), encoding="utf-8") as f:
        linhas = [l.rstrip("\n") for l in f if l.strip()]
    random.Random(42).shuffle(linhas)
    return linhas[:n]

treino = carregar("train.txt", N_AMOSTRAS)
val    = carregar("val.txt", 2_000)
print(f"{len(treino):,} linhas de treino | exemplo:\n  {treino[0][:160]}")

from datasets import Dataset
ds_treino, ds_val = Dataset.from_dict({"texto": treino}), Dataset.from_dict({"texto": val})

# %% [3] Tokenização + packing (blocos de 1024 tokens) -------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODELO)
tok.pad_token = tok.pad_token or tok.eos_token

def packer(linhas, bloco=SEq):
    """Concatena todas as linhas tokenizadas e fatia em blocos de `bloco` tokens.
    O Trainer de CLM recebe labels=input_ids e o próprio modelo faz o shift."""
    ids, buf = [], []
    for l in linhas:
        buf.extend(tok(l, add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= bloco:
            ids.append(buf[:bloco]); buf = buf[bloco:]
    if buf: ids.append(buf + [tok.pad_token_id] * (bloco - len(buf)))
    return np.array(ids, dtype=np.int32)

print("empacotando treino...")
X = packer(treino)
ds_treino = Dataset.from_dict({"input_ids": X, "labels": X.copy()})
print("validação...")
Xv = packer(val)
ds_val = Dataset.from_dict({"input_ids": Xv, "labels": Xv.copy()})
print(ds_treino, ds_val)

# %% [4] Modelo 4-bit + LoRA ----------------------------------------------------
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
modelo = AutoModelForCausalLM.from_pretrained(
    MODELO, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
modelo.config.use_cache = False

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
modelo = prepare_model_for_kbit_training(modelo)
peft = get_peft_model(modelo, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALFA, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
peft.print_trainable_parameters()

# %% [5] Treinamento -------------------------------------------------------------
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

args = TrainingArguments(
    output_dir="/kaggle/working/checkpoints",
    per_device_train_batch_size=4,        # T4 16 GB: 4 c/ seq 1024 em 4-bit
    gradient_accumulation_steps=8,        # lote efetivo 32
    num_train_epochs=EPOCAS,
    learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
    logging_steps=50, eval_strategy="steps", eval_steps=500,
    save_steps=1000, save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit", report_to="none", seed=42)

col = DataCollatorForLanguageModeling(tok, mlm=False)
treinador = Trainer(model=peft, args=args, train_dataset=ds_treino,
                   eval_dataset=ds_val, data_collator=col)
treinador.train()
peft.save_pretrained("/kaggle/working/lora_fretes")     # adapters (~50 MB)
tok.save_pretrained("/kaggle/working/lora_fretes")
print("LoRA salvo em /kaggle/working/lora_fretes")

# %% [6] Estimar frete (inferência) ----------------------------------------------
from transformers import pipeline
ger = pipeline("text-generation", model=peft, tokenizer=tok,
               device=0 if DEV == "cuda" else -1)

def estimar_frete(prompt, max_novos=24, temp=0.3):
    """Prompt deve terminar em '=>' — o modelo completa com o frete."""
    saida = ger(prompt, max_new_tokens=max_novos, do_sample=temp > 0,
                temperature=max(temp, 0.05), top_k=20, num_return_sequences=1)
    return saida[0]["generated_text"]

exemplo = ("FRETE maritimo importacao | ano=2024 | mes=06 "
           "| produto=OLEO DIESEL NCM=27101921 | origem=Estados Unidos | uf=RJ "
           "| peso_kg=50_000_000 | fob_usd=30_000_000 | brent=82 =>")
print(estimar_frete(exemplo))

# avaliação: MAE do frete_usd_por_kg gerado vs real (amostra da validação)
def extrair_kg(txt):
    m = re.search(r"frete_usd_por_kg=([\d.]+)", txt)
    return float(m.group(1)) if m else None

acertos, total = 0, 0
for l in val[:300]:
    contexto, alvo = l.split("=>", 1)
    real = extrair_kg("=>" + alvo)
    if real is None: continue
    gerado = extrair_kg(estimar_frete(contexto.strip() + " =>", temp=0.2) or "")
    if gerado is not None:
        total += 1
        acertos += abs(gerado - real)
print(f"\nMAE frete_usd_por_kg: {acertos/total:.4f} US$/kg em {total} amostras")

# %% [7] (OPCIONAL) Baseline tabular LightGBM — comparação com modelos TabArena
# TabArena/TabPFN resolvem previsão numérica; este baseline mostra o
# desempenho "clássico" com as mesmas features do corpus.
!pip -q install lightgbm
from lightgbm import LGBMRegressor
import pandas as pd

def parse_linha(l):
    """contexto (campos 'k=v' separados por '|') + alvo (após '=>')."""
    d = {}
    contexto, _, alvo = l.partition("=>")
    for parte in contexto.split("|"):
        if "=" in parte:
            k, v = parte.split("=", 1)
            d[k.strip()] = v.strip()
    d.update(dict(re.findall(r"(\w+)=(\S+)", alvo)))
    return d

dados = [parse_linha(l) for l in treino
         if l.startswith("FRETE maritimo importacao") and "=>" in l]
df = pd.DataFrame(dados)
df = df.dropna(subset=["frete_usd_por_kg"])
alvo = df.pop("frete_usd_por_kg").astype(float)
alvo = np.log1p(alvo)                      # frete/kg é lognormal
cat = ["produto", "origem", "uf", "ano", "mes"]
for c in cat: df[c] = df[c].astype("category")
num = ["peso_kg", "fob_usd", "brent"]
for c in num:
    df[c] = pd.to_numeric(df[c].str.replace("_", "", regex=False), errors="coerce")
df = df[cat + num].fillna(0)
m = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=127)
m.fit(df[: len(df)//10*9], alvo[: len(df)//10*9])
pred = m.predict(df[len(df)//10*9:])
mae = np.mean(np.abs(np.expm1(pred) - np.expm1(alvo[len(df)//10*9:])))
print(f"LightGBM MAE frete_usd_por_kg: {mae:.4f} US$/kg")
