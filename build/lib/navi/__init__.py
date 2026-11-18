# -*- coding: utf-8 -*-
"""navi — LLM para estimar fretes marítimos (origem Brasil).

Módulos:
  montar_dataset  dados Comex/UNCTAD/notícias → corpus textual + tokenizer BPE
  treinar_llm     GPT decoder-only do zero (PyTorch puro)
  gerar           estima frete a partir de um prompt no template do corpus

Comandos instalados: navi-montar, navi-treinar, navi-gerar.
"""

__version__ = "0.1.0"
