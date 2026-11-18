# Treinar a LLM de fretes do zero

Pipeline: `dados/` → `montar_dataset.py` → corpus textual + tokenizer BPE → `treinar_llm.py` (GPT do zero em PyTorch) → `gerar.py`.

## 🎯 Fine-tuning do INPRIORA (modelo de fundação tabular) — `kaggle_inpriora.py`

O [Inpriora](https://github.com/ai-for-data/inpriora) (Ant Group, Apache 2.0) é o modelo que aparece na frente do [TabArena](https://huggingface.co/spaces/TabArena/leaderboard) — **é tabular, não LLM**: não existe fine-tune por gradiente. A adaptação ao domínio é *in-context*: `fit(X, y)` fornece seus fretes reais como contexto e o modelo prevê novas consultas. O script:

1. Lê `fretes-corpus.zip` (anexado como dataset) e monta a matriz: `ncm, origem, uf, ano, mes, peso_kg, fob_usd, brent → frete_usd_por_kg` (em log)
2. `InprioraRegressor.fit` com 10 mil linhas de contexto + predição com **quantis** (intervalo de confiança 80% de graça)
3. Baseline LightGBM com o mesmo contexto (justo) e com 300 mil linhas (para ver o custo de escala do ICL)
4. MAE por país de origem e por ano — onde o modelo acerta/erra
5. Exemplo de estimativa de frete novo (US$/kg + intervalo)

No Kaggle: mesmo passo a passo do zip da seção abaixo; cole as células de `kaggle_inpriora.py`. Funciona em CPU ou GPU (T4 acelera).

## 🎯 Fine-tuning com GRADIENTE do TabICL v2 — `kaggle_tabicl_ft.py`

TabICL v2 (SODA/INRIA, open source, 28M params, **sem Prior Labs**) tem fine-tuning **oficial** com gradiente: `FinetunedTabICLRegressor` (perda pinball sobre os 999 quantis, AdamW + early stopping) — e prevê quantis, intervalo de confiança de graça igual ao Inpriora. O script compara 4 regimes no MESMO conjunto de teste (20 mil fretes reais):

| Regime | Treino | Como |
|---|---|---|
| A. TabICL v2 ICL | 10 mil linhas de contexto (pesos congelados) | `TabICLRegressor` |
| B. **TabICL v2 fine-tuned** | 30 mil linhas (25k fit + 5k val), **gradiente** | `FinetunedTabICLRegressor` |
| C. LightGBM | mesmas 30 mil | gradiente clássico |
| D. LightGBM | 300 mil | referência de escala |

Respostas: B vencer A = ganho real do gradiente; B encostar em D = modelo de fundação compensa a escala. Cobertura do intervalo 80% de A e B, MAE por país de origem e por ano (desvio temporal). Zero-shot validado localmente em CPU. Avisos conhecidos: o TabICL codifica categorias ordinalmente e reclama de cardinalidade > 40 (ncm/origem) — funciona mesmo assim; se quiser espremer mais, serializar strings com `skrub.TableVectorizer`. Fallback de modelos: `finetune_mode` alternativos via TabTune se quiser testar outros (TabFM, Mitra, Orion...).

## 🚀 Treinar no Kaggle com LLM aberta (Qwen) — `kaggle_finetune.py`

Além do treino do zero acima, dá para **fazer fine-tuning de uma LLM aberta** (Qwen2.5-1.5B) com QLoRA em cima do mesmo corpus — melhor ponto de partida que treinar do zero. Código pronto em **`kaggle_finetune.py`** (células para colar num notebook Kaggle com GPU).

Passo a passo:
1. Faça upload de `treino/fretes-corpus.zip` como **dataset Kaggle** ("fretes-corpus", ~38 MB)
2. Crie um Notebook no Kaggle, **anexe esse dataset como Input** e **ative a GPU** (T4 ×2 ou P100)
3. Cole as células de `kaggle_finetune.py` e rode Run All
   - fine-tuning QLoRA do Qwen2.5-1.5B: ~2–4 h para 1 época de 200 mil amostras
   - saída: `lora_fretes/` (adapters LoRA) em `/kaggle/working` → baixe o arquivo
4. A célula final inclui **baseline LightGBM** para comparar MAE numérico

Sobre o [TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard): é ranking de **modelos tabulares** (TabPFN v2 etc.) — úteis para previsão numérica, sem fine-tuning de linguagem. A LLM aberta certa para fine-tune é Qwen/Llama/Mistral; o baseline LightGBM da célula [7] faz o papel tabular.

## Pré-requisitos
```bash
python3 -m venv /tmp/venv_nav
/tmp/venv_nav/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
    tokenizers pandas pyarrow openpyxl numpy
# com GPU NVIDIA: pip install torch (versão CUDA)
```

## 1. Montar o dataset
```bash
cd ~/Área de trabalho/navi
/tmp/venv_nav/bin/python treino/montar_dataset.py \
    --n-frete 400000 --n-unctad 60000 --n-noticias 40000
```
Gera:
- `treino/corpus/train.txt` / `val.txt` — uma amostra por linha:
  - `FRETE maritimo importacao | ano=2024 | mes=08 | produto=... NCM=... | origem=... | uf=... | peso_kg=... | fob_usd=... | brent=... => frete_usd=... | frete_usd_por_kg=...`
  - idem para exportação (UNCTAD) e artigos de notícia (navegação/petróleo)
- `treino/corpus/tokenizer.json` — BPE 16k **treinado do zero** no próprio corpus

O alvo (frete) fica no **fim** da linha de propósito: o modelo aprende a completar o contexto com a estimativa.

## 2. Treinar
```bash
# smoke test (~2 min, valida o pipeline):
/tmp/venv_nav/bin/python treino/treinar_llm.py --amostras 30000 --passos 300

# treino de verdade (CPU 8 núcleos: algumas horas; ajuste --lote conforme a RAM):
/tmp/venv_nav/bin/python treino/treinar_llm.py \
    --emb 256 --camadas 6 --bloco 256 --lote 32 --passos 20000

# retomar do último checkpoint:
/tmp/venv_nav/bin/python treino/treinar_llm.py --passos 40000 --resume
```
Checkpoints em `treino/checkpoints/modelo_NNNN.pt` (a cada 500 passos e no fim). Perda de treino deve cair de ~9 para <2 nos primeiros milhares de passos; a de validação acompanha enquanto o modelo ainda está subajustado (modelo pequeno + corpus grande = pouco risco de overfitting).

Config padrão: 6 camadas, emb 256, 8 cabeças ≈ **7,8 M parâmetros** — treina em CPU. Com GPU, suba para `--emb 384 --camadas 8 --bloco 512 --lote 64`.

## 3. Estimar frete
```bash
/tmp/venv_nav/bin/python treino/gerar.py \
"FRETE maritimo importacao | ano=2025 | mes=03 | produto=OLEO BRUTO DE PETROLEO NCM=27090010 | origem=Arabia Saudita | uf=RS | peso_kg=500_000 | fob_usd=250_000 | brent=85 =>"
```
O modelo completa: `frete_usd=... frete_usd_por_kg=...` — esse número é a estimativa.

Exemplos de prompt exportação (dados UNCTAD):
```
FRETE maritimo exportacao Brasil | ano=2020 | produto=soja | destino=China | custo_pct_fob=9.5 =>
```

## Dicas
- **Conheça a distribuição**: o modelo aprende a distribuição condicional completa, não só a média. Rode o gerar com `--temp 0.3` para estimativa "consenso" e temperaturas maiores para ver cenários.
- O prompt deve seguir **exatamente** o template do corpus (mesmos campos, mesma ordem) — é assim que a LLM foi treinada.
- Para melhorar: mais passos; mais amostras (`--n-frete`); adicionar features (HARPEX/FBX defasados, taxa de ocupação ANTAQ) no template; aumentar o modelo.
- Um regressor clássico (LightGBM) sobre os meses/rotas costuma acertar melhor numericamente que uma LLM pequena; a vantagem da LLM é o contexto livre (notícias, rotas novas, texto).

## Arquivos
| Arquivo | Papel |
|---|---|
| `montar_dataset.py` | dados → corpus + tokenizer |
| `treinar_llm.py` | GPT decoder-only puro PyTorch, do zero |
| `gerar.py` | carrega checkpoint e estima frete |
