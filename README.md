# navi — LLM para estimar fretes marítimos (origem Brasil)

Treinar modelos de linguagem para **estimar o custo de frete marítimo** de importação/exportação
usando fretes reais declarados à Receita Federal (Comex Stat/SECEX), tarifas de mercado
(BDI, HARPEX, FBX, Brent) e notícias de navegação.

## Estrutura

```
dados/                 bases baixadas (7,4 GB — NÃO entram no repo; guia de fontes abaixo)
  └─ LEIA-ME.md        ⭐ guia completo: o que é cada base, como foi baixada, como usar
treino/                pipeline de treino (entra inteiro no repo)
  ├─ README-treino.md  ⭐ documentação do pipeline: dataset, treino, estimação
  ├─ montar_dataset.py      dados → corpus textual + tokenizer BPE (460 mil linhas)
  ├─ treinar_llm.py         GPT decoder-only do zero, PyTorch puro (8,9 M params)
  ├─ gerar.py               carrega checkpoint e estima frete a partir de um prompt
  ├─ kaggle_finetune.py     fine-tune QLoRA do Qwen2.5-1.5B (notebook Kaggle c/ GPU)
  ├─ kaggle_inpriora.py     adaptação in-context do Inpriora (sem gradiente) + LightGBM
  ├─ kaggle_tabicl_ft.py    fine-tune com GRADIENTE do TabICL v2 (INRIA) + LightGBM
  ├─ fretes-corpus.zip      corpus pronto (train.txt + val.txt, 40 MB — p/ dataset Kaggle)
  └─ corpus/tokenizer.json BPE 16k treinado no corpus
```

## Corpus

Cada linha é uma amostra com o alvo no fim (o modelo aprende a completar):

```
FRETE maritimo importacao | ano=2020 | mes=02 | produto=Iobitridol NCM=29242964 | origem=França | uf=RJ | peso_kg=5_593 | fob_usd=573_406 | brent=55.7 => frete_usd=1_427 | frete_usd_por_kg=0.2551
```

460 mil linhas: 400 mil fretes reais Comex 2016–21 + 40 mil exportação UNCTAD + 26 mil notícias.

## Uso rápido

```bash
# 1. montar o corpus (requer as bases em dados/ — ver dados/LEIA-ME.md)
python treino/montar_dataset.py --n-frete 400000 --n-unctad 60000 --n-noticias 40000
# (ou apenas descompacte treino/fretes-corpus.zip em treino/corpus/)

# 2. treinar (CPU ok; GPU acelera)
python treino/treinar_llm.py --emb 256 --camadas 6 --bloco 256 --lote 32 --passos 20000

# 3. estimar um frete
python treino/gerar.py "FRETE maritimo importacao | ano=2025 | mes=03 | produto=OLEO BRUTO DE PETROLEO NCM=27090010 | origem=Arabia Saudita | uf=RS | peso_kg=500_000 | fob_usd=250_000 | brent=85 =>"
```

Para os modelos tabulares (Inpriora in-context, TabICL v2 fine-tuned, LightGBM):
notebooks Kaggle prontos em `treino/kaggle_*.py` — instruções em
`treino/README-treino.md`.

## Pré-requisitos

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
    tokenizers pandas pyarrow numpy
# GPU NVIDIA: pip install torch (versão CUDA)
```

## Dados brutos

As bases completas (Comex, ANTAQ, UNCTAD, séries de tarifas, notícias) **não estão no
repo** — o guia de download/uso de cada uma está em [`dados/LEIA-ME.md`](dados/LEIA-ME.md).
