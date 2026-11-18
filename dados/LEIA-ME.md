# Bases de dados de fretes marítimos — Guia

Baixado em 17/11/2026. Total: ~10 GB comprimidos / 3+ GB extraídos + ~6 GB Comex. **Objetivo: treinar LLM para estimar fretes.**

## 7. `comex/` — Comex Stat (SECEX/MDIC) 🇧🇷⭐⭐ **A MAIS VALIOSA PARA TREINO**

**O dado de frete mais granular do Brasil**: importações declaram `VL_FRETE` (frete em US$) e `VL_SEGURO` por mês × NCM (8 díg.) × país × UF × via × unidade alfandegária!

- `imp/IMP_YYYY.csv` — 1997–2026, ~90–130 MB/ano, colunas: `CO_ANO, CO_MES, CO_NCM, CO_UNID, CO_PAIS, SG_UF_NCM, CO_VIA (via=1 marítimo), CO_URF, QT_ESTAT, KG_LIQUIDO, VL_FOB, VL_FRETE, VL_SEGURO`
- `exp/EXP_YYYY.csv` — exportações (sem VL_FRETE, mas com peso/valor por destino)
- `aux/TABELAS_AUXILIARES.xlsx` — decodificadores: NCM, país, via, URF
- Fonte: `balanca.mdic.gov.br/balanca/bd/comexstat-bd/ncm/` — separador `;`, latin-1. ⚠️ Certificado SSL do site tem cadeia incompleta: usar `--cacert` com o intermediário GlobalSign (ver histórico da sessão). Arquivos inexistentes devolvem PHP (HTTP 206/200).

**Com isso dá para calcular US$/kg frete real declarado** por rota Brasil↔país × produto — target direto para a LLM.

## 8. `indices/` — Séries temporais de tarifas (para janelas de treino)

| Arquivo | Série | Período |
|---|---|---|
| `bdi_1985_2013.csv` | Baltic Dry Index diário | 01/1985–06/2013 (7.350 pontos, GitHub) |
| `harpex_2021_2026.csv` | HARPEX afretamento contêiner (semanal) | 02/2021–10/2026 (255 pontos, costurado de snapshots do site Harper Petersen) |
| `gscpi_fred.csv` | Global Supply Chain Pressure Index (mensal) | 1998–2024 (FRED) |
| `container-freight-rates.csv` | FBX global + China→USWC + China→N.Europa (semanal) | recente |
| `fbx_todas_rotas.csv` | 13 índices FBX atuais | snapshot |
| `bunker_prices.csv` | VLSFO Singapore (US$/t) | recente |
| `../unctad/csv/US_CommodityPrice*_*.csv` | Preços mensal/anual de commodities (petróleo, combustível, carvão...) da UNCTAD — proxy de bunker | anos 2000–2023 |

⚠️ **Gap conhecido: BDI 2013–2026.** Stooq/investing.com bloqueiam download programático; caminhos: conta gratuita em investing.com, relatórios semanais do Baltic Exchange (site com challenge), ou série paga. O HARPEX cobre contêiner pós-2021; UNCTAD GTCDIT cobre fretes implícitos 2016–2021.

## 9. `faostat/` — Matriz de comércio agrícola mundial (FAO)

- `Trade_DetailedTradeMatrix_E_All_Data.zip` (409 MB) — fluxos bilaterais de todos os países, todos os itens agrícolas, **quantidade (t) e valor (US$)**, 1986–2024
- `comercio_agricola_brasil.csv` (20 MB, 77.672 linhas) — já filtrado: Brasil ↔ 230 parceiros por produto/ano/direção
- Licença CC BY 4.0. Fonte: bulks-faostat.fao.org

## 10. `ais/` — Trincas data-navio-porto global (AIS 2020)

- `triplet_2020.zip` (31 MB) — Zenodo 10783698: cada associação navio×porto com data, para reconstruir viagens. Combinável com LNG-T3.

## 11. `unlocode/` — Códigos UN/LOCODE de portos

- `code-list.csv` — 116.477 locais (portos, terminais) com coordenadas — para geolocalizar origens/destinos e calcular distâncias
- Espelho GitHub (datahub `datasets/un-locode`); original UNECE bloqueia bots (403)

## 12. `navios/` — Registro mundial de navios (Wikidata)

- `wikidata_navios.csv` — **31.940 navios** com nome, **IMO**, tipo (instância de), bandeira, dono, operador, comprimento, tonelagem, ano de construção. Fonte: query.wikidata.org (propriedade P458), CC0.
- Complementos: `lng-t3/LNG_tanker.csv` (frota GNL com detalhes), `antaq/Atracacao/` (IMO de 1,3 M de atracações nos portos BR)

## 13. `noticias/` — Notícias para contexto da LLM

| Arquivo | Conteúdo |
|---|---|
| `shipping_news_articles.parquet` (227 MB) | **40.013 artigos completos** de notícias de navegação (título, data, texto + embeddings + classes) |
| `shipping_news_summary.parquet` (113 MB) | Os mesmos 40 mil artigos com **resumos** (LED/BART) |
| `financial_news.parquet` (136 MB) | **1.845.559 manchetes** financeiras (headline, veículo, data, ação) |
| `crude_oil_news.csv` (52 MB) | 5.937 notícias do mercado de petróleo com texto completo, sentimento e resumo IA |
| `gdelt_noticias_frete_12m.csv` | Varredura GDELT: metadados (título/veículo/data/país) de notícias sobre frete, Red Sea, Panamá, congestionamento, tanker, bunker — últimos 12 meses |
| `oil_news_*.csv` | Agregados de notícias de petróleo (países, organizações, mensal) |

Preços de petróleo: `indices/brent_diario.csv` (1987–2026) e `indices/wti_diario.csv` (1986–2026) via FRED — covariável essencial (bunker).

---

## Estado dos downloads ao encerrar esta rodada

- `comex/imp/` — ✅ **COMPLETO**: 30 anos (1997–2026), 3,2 GB, todos com `VL_FRETE`/`VL_SEGURO`. Em 2024: 1,06 M importações marítimas c/ frete = US$ 11,87 bi, média US$ 0,069/kg
- UNCTAD `US_TradeMatrix_Part1-5` — Wayback Machine bloqueou por volume (retry depois); matriz mundial já coberta por FAOSTAT (agrícola) + GTCDIT (fretes)
- OECD MTC (1991–2007) — dataset descontinuado da API OECD, sem mirror livre
- FTRI da UNCTAD = "Facilitation of Trade" (índice de facilitação), **não** é índice de tarifas

## Sugestão de pipeline de treino

1. **Target**: `VL_FRETE/KG_LIQUIDO` (US$/kg) e/ou frete ad-valorem do Comex IMP (2016–2026, marítimo) + UNCTAD GTCDIT (2016–2021, marítimo, todos os países)
2. **Features de rota**: origem/destino (UNLOCODE + distância), produto (NCM→HS/CPC), volume, TEU
3. **Features de mercado**: BDI/HARPEX/FBX defasados, bunker/commodity prices, GSCPI
4. **Features de porto**: ANTAQ atracação (tempos de espera = congestionamento), port calls UNCTAD, LSCI/PLSCI (conectividade)

## Limitações conhecidas

---

## 1. `unctad/` — UNCTAD Trade-and-Transport Dataset (GTCDIT) ⭐

**O que é:** custo de frete internacional por origem → destino → produto (HS) → modo de transporte, derivado da diferença CIF–FOB (Comtrade). 170+ economias, ~87% do comércio global. Licença CC BY 3.0 IGO.

**Período:** 2016–2021 · **Modo marítimo = código 21**

| Arquivo | Conteúdo |
|---|---|
| `csv/US_TransportCosts_Sea_Main_ST202101131757_v1.csv` (1,5 GB) | **Fretes marítimos**: custo do frete (US$), % sobre valor FOB, US$/kg, US$/kg por 10.000 km |
| `csv/US_TransportCosts_All_Detail_ST202101141643_v1.csv` (1,6 GB) | Detalhe completo todos os modos: valor CIF, valor FOB, quantidade |
| `csv/US_SeaborneTrade_*.csv` | Comércio marítimo mundial por país |
| `csv/US_PortCalls*.csv`, `US_MerchantFleet_*.csv`, `US_LSCI_*.csv`, `US_FTRI_*.csv`, `US_PLSCI_*.csv`, `US_ContPortThroughput_*.csv`, `US_FleetBeneficialOwners_*.csv` | Escalas de navios, frota mercante, índices de conectividade (LSCI/PLSCI), índice de tarifas de fretes (FTRI) |

**Colunas do frete:** `Year, Origin, Destination, HS2012Product, TransportMode, Transport costs, Transport costs to FOB value, Transport costs per unit, Transport costs per unit per 10 000 km` + rodapés marcando valores *"Estimated"* (estimados por modelo — filtrar quando precisar de só valores reportados).

### Arquivos já filtrados para o Brasil 🇧🇷
- `fretes_maritimos_origem_brasil.csv` — **82.496 linhas** (origem Brasil, marítimo, 2016–2021)
- `fretes_maritimos_destino_brasil.csv` — **60.472 linhas** (destino Brasil)

Exemplo: `2016, Brasil → Albânia, carne bovina congelada (020230), frete US$ 46.351, 9,66% do FOB, US$ 0,141/kg`

Fonte original: [UNCTADstat Transport Costs](https://unctadstat.unctad.org/EN/TransportCost.html). Os arquivos bulk saíram do ar; estas cópias vieram do Wayback Machine.

---

## 2. `antaq/` — ANTAQ Estatístico Aquaviário (espelho Hugging Face) ⭐🇧🇷

**O que é:** TODOS os registros de movimentação de carga e atracação nos portos brasileiros, 2010–2026, ~190 milhões de linhas. O portal oficial (`estatistica.antaq.gov.br`) está **offline desde jun/2026** (403 Cloudflare) — este espelho é de [huggingface.co/datasets/vinicius-souza/antaq](https://huggingface.co/datasets/vinicius-souza/antaq) (arquitetura medalhão, licença ODbL dos dados originais).

| Pasta | Linhas | Conteúdo |
|---|---|---|
| `Carga/` | 31,6 M | Cada carga: **Origem, Destino** (código porto/instalação), mercadoria (HS2/HS4), natureza, sentido, **TEU, peso bruto (t)**, flags longo curso/cabotagem |
| `Carga_Conteinerizada/` | 142,8 M | Carga por contêiner (peso por mercadoria por caixa) |
| `Atracacao/` | 1,32 M | Cada atracação: **Nº do IMO do navio**, porto, berço, armador/nacionalidade, datas chegada/atracação/desatracação |
| `TemposAtracacao/` | 1,32 M | Tempos de espera/operação/estadia |
| `Carga_Hidrovia/`, `Carga_Regiao/` | 13,3 M | Visões por hidrovia e região |
| `gold/` | — | Tabelas analíticas prontas (`atracacao_master` com 42 colunas junta tudo: navio + porto + tempos + peso + TEU) |

**Chave de junção:** `Carga.IDAtracacao → Atracacao.IDAtracacao` (assim se descobre o navio que levou cada carga).

Exemplo (2024, longo curso): Itaqui (BRMA002) → Tánger Med (CNTAO), 126,3 M t.

---

## 3. `lng-t3/` — Comércio global de GNL por AIS (Zenodo, CC BY 4.0)

Viagens de navios-tanque GNL 2020–2024, resolução diária:
- `LNG_tanker_voyage.csv` — cada viagem: **IMO, terminal/porto/país origem → destino, volume (m³), distância**
- `LNG_trade_daily.csv` — fluxo país-a-país diário
- `LNG_tanker.csv` — frota (406 navios) · `LNG_terminal.csv` — 545 terminais
- DOI: [10.5281/zenodo.19571058](https://zenodo.org/records/19571058)

---

## 4. `fbx/` — Índices de frete de contêiner (FBX / Baltic)

- `container-freight-rates.csv` — histórico semanal recente (FBX global + rota China→USWC + China→Norte Europa, US$/FEU), do [Forwardernews](https://www.forwardernews.com/trackers/container-freight-rates/data.csv)
- `fbx_todas_rotas.csv` — valor atual das 13 rotas FBX00–FBX26 ([BOP Data](https://data.bop.ink/thuong-mai-hai-quan/cuoc-chi-so-van-tai-quoc-te/freightos-baltic-index))
- Histórico completo desde 2016 requer conta no [Baltic Exchange](https://www.balticexchange.com/en/data-services/market-information0/container-services.html) / API da Freightos

---

## 5. `gofi/` — Rotas de fretes de grãos (IGC GOFI)

- `routes.csv` — pares origem → destino porto por carga (milho, trigo etc.), referência de rotas do índice de fretes de grãos. Licença CC BY.

---

## 6. `ipea/` — Séries históricas ANTAQ via Ipea

- `cabotagem_qde.csv` — cargas de cabotagem 1921–2024 (série anual histórica)
- `cargas_mensal.csv` — cargas aquaviárias mensais 2020–2026

---

## Como usar rápido

```bash
# venv com pyarrow/pandas
/tmp/venv_nav/bin/python

# Fretes marítimos saindo do Brasil (UNCTAD, 2016-2021)
import pandas as pd
df = pd.read_csv('unctad/fretes_maritimos_origem_brasil.csv')

# Cargas nos portos BR 2024 + navio de cada carga
c = pd.read_parquet('antaq/Carga/2024.parquet')
a = pd.read_parquet('antaq/Atracacao/2024.parquet')
m = c.merge(a, on='IDAtracacao')
```

## Limitações conhecidas

- UNCTAD: frete estimado por CIF–FOB (não é tarifa de mercado); valores estimados marcados como "Estimated"
- ANTAQ: espelho não-oficial (dados originais ODbL); não há valor de frete em reais — o BIF oficial (que tinha preço) está offline com o portal
- UNCTAD GTCDIT para no ano de 2021; ANTAQ cobre até 2026
- Preço de frete por rota em nível de mercado (não estimado) = pago (Drewry WCI, Xeneta, Platts)
