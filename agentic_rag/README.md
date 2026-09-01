# Agentic RAG — Türkçe Medikal Doküman Sistemi

TAMD ("Türkçe Medikal Dokümanlar") adlı Medikal LLM (TUS veri seti üzerinde fine‑tune edilmiş) için **agentic RAG** bileşenleri. Sistem, TUSDATA/MinerU'dan gelen Markdown dökümanları hiyerarşik parçalara ayırır (chunk), bunları Qdrant vektör veritabanına indeksler ve LangGraph tabanlı **Responder → Pruner → Compactor** ajan döngüsüyle hekimlere atıf yapılabilir, kanıta dayalı cevaplar üretir.

---

## Dizin Yapısı (`ls -R`)

```
├── CLAUDE.md                          Agent geliştirme yönergeleri (LLM davranış rehberi)
├── README.md                          İşbu dosya
├── requirements.txt                   Python bağımlılıkları
├── main.ipynb                         LangGraph ajan grafiğini çalıştıran Jupyter notebook'u
├── data/                              Ham Markdown dokümanlar (branş bazında klasörler)
│   ├── ANATOMİ/                       ANATOMİ 1.pdf.md, ANATOMİ 2.pdf.md
│   ├── BİYOKİMYA/                     BİYOKİMYA 1.pdf.md, BİYOKİMYA 2.pdf.md
│   ├── DAHİLİYE/                      DAHİLİYE 1.md, DAHİLİYE 2.md, DAHİLİYE 3.md
│   ├── FARMAKOLOJİ/                   FARMAKOLOJİ 1.md, FARMAKOLOJİ 2.md
│   ├── FİZYOLOJİ-HİSTOLOJİ-EMBRİYOLOJİ/  FİZYOLOJİ 1.md, FİZYOLOJİ-HİSTOLOJİ-EMBRİYOLOJİ 1.md
│   ├── GENEL CERRAHİ/                 GENEL CERRAHİ 1.md, GENEL CERRAHİ 2.md
│   ├── KADIN DOĞUM/                   KADIN DOGUM 1.md, KADIN DOĞUM 2.md
│   ├── KÜÇÜK STAJLAR/                 KÜÇÜK STAJLAR 1.md, KÜÇÜK STAJLAR 2.md
│   ├── MİKROBİYOLOJİ/                 MİKROBİYOLOJİ 1.md, MİKROBİYOLOJİ 2.md
│   ├── PATOLOJİ/                      PATOLOJİ 1.md, PATOLOJİ 2.md
│   └── PEDİATRİ/                      PEDİATRİ 1.md, PEDİATRİ 2.md, PEDİATRİ 3.md
├── scripts/
│   ├── __init__.py                    Paket işaretçisi
│   ├── agents/
│   │   ├── responder_agent.py         Responder düğümü: retrieve_documents çağırır, sentez yapar
│   │   ├── pruning_agent.py           Pruner düğümü: çekilen dokümanı sorguyla ilgili kısma budar
│   │   └── compaction_agent.py        Compactor düğümü: tur sonunda araç mesajlarını özete indirir
│   ├── functions/
│   │   ├── tools.py                   create_retrieve_documents_tool: Retrieval aracını üretir
│   │   └── tool_contents.py           TODO: Retriever başlatıp doğrudan çağıran yardımcı
│   ├── llm/
│   │   └── __init__.py                LLM bağlayıcıları (taslak)
│   ├── prompts/
│   │   └── prompts.py                 RESPONDER ve PRUNING sistem prompt şablonları
│   ├── retrieval/
│   │   ├── data_loader.py             Markdown dosyalarını Document nesnelerine yükler
│   │   ├── document_chunker.py        Hiyerarşik, breadcrumb'lı parçalayıcı (3 aşamalı)
│   │   ├── main_retriever.py          CLI: yükle → parçala → Qdrant'a indeksle
│   │   ├── qdrant_vector_store.py     Lokal Qdrant depolama/cevri (LocalQdrantVectorStore)
│   │   └── retrieval_system.py        Eski/alternatif BM25 + reranker sistemi (not: chunk'a bağımlı)
│   └── workflow/
│       └── graph.py                   LangGraph StateGraph: responder → retrieval_tools → prune → compact
└── docs/                              Mimar. diyagramları (.puml, .svg) — aşağıya bakın
```

> Not: Yüklenecek tüm Markdown'lar `data/` altında; sadece `*.md` uzantılılar okunur.

---

## Retrieval Pipeline — Nasıl Çalışır

```
data/*.md
   │  data_loader.load_documents()
   ▼
Document (metadata: source, category, title)
   │  document_chunker.chunk_documents()
   ▼
Hiyerarşik & breadcrumb'lı parçalar
   │  main_retriever: LocalQdrantVectorStore.build()
   ▼
Qdrant koleksiyonu "turkish_medical_documents"
   │  LocalQdrantVectorStore.as_retriever(k=6)
   ▼
VectorStoreRetriever (sorgu → en yakın 6 chunk)
```

`main_retriever.py` bu zincirin tamamını tek komutla çalıştırır:

```bash
# agentic_rag/ dizini içinden (scripts paketi orada olduğu için):
nohup python3 -u -m scripts.retrieval.main_retriever > retrieval_indexing.log 2>&1 &
```

- `nohup` / `&` → oturum kapanınca süreç ölmez; log `retrieval_indexing.log`'a akar.
- `-u` → Python çıktısı buffering yapmadan (gerçek zamanlı) yazar.
- Eğer `.venv` kullanıyorsan: `nohup .venv/bin/python3 -u -m scripts.retrieval.main_retriever > retrieval_indexing.log 2>&1 &`

Çıktı: `Embedding modeli intfloat/multilingual-e5-base` ile her chunk vektörleştirilir (CPU, normalize), Qdrant lokalde persist edilir.

---

## document_chunker.py — 3 Aşamalı Hiyerarşik Parçalama

**Neden?** MinerU/TUSDATA çıktıları düz karakter sayısına göre parçalansa; bir hastalığın "Komplikasyonlar" başlığı bir chunk'ta, ait olduğu `# Crohn Hastalığı` bir önceki chunk'ta kalır. Bu, cevap sentezi sırasında **bağlamın kopmasına** ve **eksik/hatalı tıbbi atfa** yol açar. Data_loader'ın `category`'si zaten branşı (örn. `DAHİLİYE`) taşır; chunker bunu doküman içi `#`/`##`/`###` başlıklarıyla birleştirerek her parçaya tam bir konu yolu ekler.

### Aşama 1 — `normalize_markdown_headers(text)`

MinerU bazen `1.1`, `A)`, `2)` gibi **alt maddelere** ana başlıkla aynı sayıda `##` hashi atar. Regex, yalnızca `#`/`##` seviyesindeki alt madde satırlarını tespit edip bir seviye (`###`) aşağı indirir; gerçek üst başlıklar (`## 1. KALP YETMEZLIĞI` gibi) ve zaten doğru hizalanmış `###` satırları korunur.

| Girdi | Çıktı |
|-------|-------|
| `## 1.1 Tanım` | `### 1.1 Tanım` |
| `## A) Atriyal fibrilasyon` | `### A) Atriyal fibrilasyon` |
| `## 2) Hipertansiyon` | `### 2) Hipertansiyon` |
| `## 1. KALP YETMEZLIĞI` | değişmez (gerçek bölüm) |
| `## • Betimleyici satır` | değişmez (madde işareti, başlık değil) |

### Aşama 2 — İki Kademeli Bölme

1. **`MarkdownHeaderTextSplitter`** (`#`, `##`, `###` sınırlarında `strip_headers=False`) metni mantıksal bölümlere ayırır; her bölümün başlığı `section`/`heading`/`subheading` metadata olarak tutulur.
2. **`RecursiveCharacterTextSplitter`** (`chunk_size=600`, `chunk_overlap=100`) bu maksimum boyutu aşan (uzun/tablo yoğun) bölümleri bağlam örtüşmesiyle yeniden boyutlandırır; altında kalan bölümler aynen korunur.

### Aşama 3 — Breadcrumb (Konu Yolu) Metadata

Her parça şu metadata'ları taşır:

```python
{
  "source":     "DAHİLİYE/DAHİLİYE 1.md",
  "category":   "DAHİLİYE",
  "title":      "DAHİLİYE 1",
  "section":    "HEMATOLOJI",          # # seviyesi
  "heading":    "DEMIR EKSIKLIĞI ANEMISi",  # ## seviyesi
  "subheading": "Laboratuvar",              # ### seviyesi (varsa)
  "breadcrumb": "DAHİLİYE > HEMATOLOJI > DEMIR EKSIKLIĞI ANEMISi",
  "chunk":      3,                    # kaynak başına sıra (geriye uyumlu)
}
```

`breadcrumb`, **Agent B**'nin hekime yapacağı atıf (grounded citation) için ve **Agent A**'nın metadata filtrelemesi için zorunlu; `chunk` anahtarı, eski `retrieval_system.flatten_context` ile bozulmadan çalışabilmek için korunur.

### Fonksiyon imzaları

```python
def normalize_markdown_headers(text: str) -> str
def chunk_documents(documents: list[Document], chunk_size=600, chunk_overlap=100) -> list[Document]
```

`chunk_documents` girdi/çıktı tipini değiştirmez; `data_loader → chunker → qdrant_vector_store` zinciri aynen çalışır.

---

## Ajan Grafiği (Agentic Workflow)

`scripts/workflow/graph.py` (`build_graph`) şu LangGraph akışını kurar:

```
START
  │
  ▼
responder ──tool_calls──▶ retrieval_tools ──▶ prune ──▶ responder
  │                                                    │
  └──────────(tool yok / limit)──────────────────────▶ compact ──▶ END
```

- **Responder** (`responder_agent.py`): Selamlaşma/sohbet dışı her bilgi sorusunda önce `retrieve_documents` aracını çağırır, sonra çekilen dokümanlara dayalı sentez yapar. Aynı turda en fazla `MAX_RETRIEVALS=1` kez arama yapar (`route_from_responder`).
- **Retrieval tools** (`tools.py`): `compressed_retriever.invoke(query)` ile en yakın chunk'ları alıp LLM'e düz metin olarak döndürür.
- **Pruner** (`pruning_agent.py`): Çekilen dokümanı yalnızca sorguyla ilgili kısma budar; gevezelik shared context'e girmez.
- **Compactor** (`compaction_agent.py`): Tur bitince o turun araç çağrısı + doküman mesajlarını tek `[Retrieved documents for: …]` özetine indirir; sonraki turlar geçmişi yeniden ödemez.

Prompt şablonları `scripts/prompts/prompts.py` içindedir (`RESPONDER_SYSTEM_PROMPT_TEMPLATE`, `PRUNING_SYSTEM_PROMPT_TEMPLATE`).

---

## Agent A / Agent B Rol Dağılımı

| Ajan | Görev | Çıktı |
|------|-------|-------|
| **Agent A (Retriever / Search)** | Qdrant'tan metadata filtreleme ve semantik arama; `breadcrumb` ile konuyu daraltır | Top-6 breadcrumb'lı chunk |
| **Agent B (Responder / Synthesis)** | Prune edilmiş chunk'lardan, orijinal başlık yolunu (breadcrumb) atıflayarak cevap üretir | Hekime atıf yapılabilir cevap |

---

## Dosya Bazında Görev Tanımları

| Dosya | Rol |
|-------|-----|
| `scripts/retrieval/data_loader.py` | `data/*.md`'yi `Document` nesnesine çevirir; `source/category/title` metadata'sını set eder. |
| `scripts/retrieval/document_chunker.py` | Markdown hiyerarşisine göre parçalar; `breadcrumb` + `chunk` metadata'sını ekler. |
| `scripts/retrieval/main_retriever.py` | Komut satırı girişi: yükle→parçala→embed→Qdrant'a yaz; kullanılabilir retriever döner. |
| `scripts/retrieval/qdrant_vector_store.py` | Lokal Qdrant'ta `turkish_medical_documents` koleksiyonunu kurar/bağlar; `as_retriever(k=6)`. |
| `scripts/retrieval/retrieval_system.py` | Eski nesil BM25 + cross‑encoder reranker (Practicus/HF) sistemi; `chunk` metadata'sına bağımlı. |
| `scripts/workflow/graph.py` | Ajan düğümlerini LangGraph StateGraph ile bağlayıp compile eder. |
| `scripts/agents/responder_agent.py` | Hedef yanıtlayıcı düğüm; retrieval kararını `route_from_responder` ile verir. |
| `scripts/agents/pruning_agent.py` | Doküman gürültüsünü azaltan budayıcı düğüm. |
| `scripts/agents/compaction_agent.py` | Uzun geçmişi kısaltan özetleyici düğüm. |
| `scripts/functions/tools.py` | `retrieve_documents` tool factory'si (retriever'a bağlı closure). |
| `scripts/prompts/prompts.py` | Sistem prompt şablonları (sentez + budama yönergeleri). |
| `main.ipynb` | Not defteri: secrets/proxy ayarları, vLLM bağlantısı, grafik kurulumu, sohbet testi. |
| `requirements.txt` | langgraph, langchain‑core, langchain‑huggingface, langchain‑qdrant, langchain‑text‑splitters, vs. |

> Uyarı: `main.ipynb` eski API adlarını (`online_inference`, `BuildRetriever`, `transfer_to_retriever`) ve `config/vllm_config.json`, `.env.example`, `scripts/llm/online_inference.py` dosyalarını referans eder. Bu dosyalar şu anda repoda yoktur; notebook bütünleşik çalıştırılmadan önce bu bağımlılıkların tamamlanması gerekir.

---

## Mimar. Diyagramları

- `docs/agentic-rag-architecture.puml` — act/component PlantUML diyagramı (chunker pipeline + ajan grafiği).
- `docs/agentic-rag-architecture.svg` — aynı diyagramın render edilmiş hali.

`.puml` dosyasını SVG'ye çevirmek için:

```bash
# PlantUML kuruluysa (java gerektirir):
plantuml -tsvg docs/agentic-rag-architecture.puml
# veya docker ile:
docker run --rm -v "$(pwd)":/docs plantuml/plantuml -tsvg /docs/docs/agentic-rag-architecture.puml
```

---

## Kurulum ve Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# İndekslemeyi arka planda başlat:
nohup python3 -u -m scripts.retrieval.main_retriever > retrieval_indexing.log 2>&1 &
tail -f retrieval_indexing.log
```