# Vectorless RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange?style=for-the-badge)](https://ollama.com)
[![BM25](https://img.shields.io/badge/Retrieval-BM25-brightgreen?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)
[![No Vectors](https://img.shields.io/badge/Vectors-None-red?style=for-the-badge)](#why-no-vectors)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Part 4 of a hands-on AI engineering series** — building in public, month by month.
>
> ← [Part 3 — Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph) &nbsp;|&nbsp; Part 5 — coming May 2026

**A fully local RAG pipeline that replaces the embedding + vector-store layer entirely with BM25 — the same battle-tested ranking algorithm that powers production search engines.**

---

## What This Repository Is

The majority of RAG implementations follow an identical pattern: generate dense embeddings, push them into a vector database, retrieve by cosine similarity. This works well — but it carries real costs: GPU or CPU time for embedding, a running vector store process, and a startup penalty on every restart as embeddings are reloaded or re-generated.

This project deliberately removes all of that.

**Vectorless RAG** replaces dense retrieval with [BM25 (Okapi BM25)](https://en.wikipedia.org/wiki/Okapi_BM25) — a probabilistic keyword ranking algorithm with decades of production use in search engines. The BM25 index is built once, persisted to disk as a lightweight `.pkl` + `.json` pair, and reloaded in milliseconds on every subsequent run. No embedding model, no vector database, no GPU dependency.

The goal is not to claim BM25 is universally superior to dense retrieval — it is not. The goal is to demonstrate that retrieval quality for precision-terminology documents can be achieved with a fraction of the infrastructure, and to make every stage of the RAG pipeline explicit and replaceable.

> Drop in your own PDFs or text files, ask questions in plain English, and get grounded answers — no vector store required.

---

## Live Demo

```
=== Vectorless RAG Pipeline ===

BM25 index loaded from disk — 312 chunks (skipping re-indexing).

Ollama model : llama3.2
Type a question or 'quit' to exit.

Question: What are the daily cleaning steps for the coffee machine?

Top 5 retrieved chunks:
  1. [score 14.32] Breville_Barista_Express.pdf p.22 — The daily cleaning routine starts with…
  2. [score 11.87] sample_coffee_machine.txt p.? — Rinse the portafilter and basket after…
  3. [score  9.54] Breville_Barista_Express.pdf p.23 — Wipe the steam wand immediately after…

Generating answer with Ollama …

Answer:
The daily cleaning steps include: 1) Rinse the portafilter and group
head after each use. 2) Purge the steam wand before and after steaming
milk. 3) Wipe the drip tray at the end of each day...
```

---

## Why No Vectors?

| | Traditional RAG | **Vectorless RAG** |
|---|---|---|
| Retrieval method | Dense cosine similarity | BM25 keyword scoring |
| Embedding model | Required (384–1536 dims) | **None** |
| Vector database | ChromaDB / Pinecone / etc. | **None** |
| Index size | Large (float arrays per chunk) | Small (inverted index) |
| First-run time | Minutes (embedding batch) | **Seconds** |
| GPU / RAM pressure | High | **Minimal** |
| Exact keyword match | Weak | **Strong** |
| Semantic similarity | Strong | Weaker |
| Works fully offline | Yes (with local models) | **Yes** |
| Restart penalty | Re-embed or re-load floats | Millisecond `.pkl` reload |

**BM25 is the right choice when:**
- Documents use precise, domain-specific terminology — manuals, legal texts, medical records, technical specifications
- Startup latency matters — no re-embedding penalty on every restart
- Hardware is constrained — runs comfortably on any machine that can run Python
- The objective is to understand RAG fundamentals without delegating retrieval to a hosted service

---

## Pipeline Architecture

```
Your Documents (PDFs / .txt)
        │
        ▼
┌───────────────────────┐
│    Document Loader    │  LangChain PyPDFLoader / TextLoader
│                       │  Per-page metadata tagging (source_file, file_type)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│     Text Splitter     │  RecursiveCharacterTextSplitter
│   chunk=1000 chars    │  overlap=200 — preserves cross-boundary context
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│     BM25 Indexer      │  rank-bm25 / BM25Okapi
│   (no embeddings)     │  lowercase tokenisation → inverted index
└──────────┬────────────┘
           │  persisted to data/bm25_index/ as .pkl + .json
           ▼
┌───────────────────────┐
│    BM25 Retriever     │  keyword scoring → top-K ranked chunks
│   (at query time)     │  reloaded from disk instantly on restart
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│      Ollama LLM       │  llama3.2 (or any locally pulled model)
│    (runs locally)     │  context-grounded answer generation
└───────────────────────┘
```

**On first run:** documents are loaded, chunked, and indexed with BM25. Index is saved to disk.
**On subsequent runs:** index is reloaded from disk in milliseconds — no re-processing whatsoever.

---

## Key Engineering Capabilities Demonstrated

| Area | Implementation |
|---|---|
| **BM25 retrieval** | `BM25Okapi` from `rank-bm25` — probabilistic term-frequency ranking, no embeddings |
| **Index persistence** | `pickle` for the BM25 object + `json` for the corpus — instant reload on restart |
| **Idempotent ingestion** | Index existence check on startup — documents are never re-indexed unnecessarily |
| **Document loading** | LangChain loaders for PDF and plain text with per-document metadata tagging |
| **Chunking strategy** | `RecursiveCharacterTextSplitter` — semantic boundary preservation with configurable overlap |
| **Tokenisation** | Lightweight lowercase whitespace tokeniser — fast, dependency-free, swappable |
| **Local LLM generation** | Ollama with context-grounded prompting and source citation per retrieved chunk |
| **Graceful empty-corpus handling** | Pipeline detects missing documents early and provides actionable guidance |
| **Swappable components** | Every parameter (LLM, chunk size, overlap, top-K, paths) is configurable from a single block |

---

## Design Decisions

**Why BM25Okapi over BM25Plus or BM25L?**
`BM25Okapi` is the canonical implementation and the most widely benchmarked variant. For document retrieval over technical corpora, it performs comparably to the newer variants while being the most interpretable. Swapping to `BM25Plus` requires a one-line change.

**Why `pickle` + `json` for index persistence rather than a database?**
The BM25 index is a pure Python object. Serialising it with `pickle` is the most direct path to disk with zero external dependencies. The corpus is stored separately as `json` to remain human-readable and inspectable without loading the index. Together they provide instant reload without introducing a running database process.

**Why a lightweight whitespace tokeniser instead of NLTK or spaCy?**
External tokenisers add dependency weight and latency for marginal gains in BM25 context. Lowercase whitespace splitting is sufficient for term-frequency ranking over technical documents and keeps the pipeline dependency-free at the retrieval layer. A more sophisticated tokeniser is a direct drop-in replacement in the `_tokenise` function.

**Why `RecursiveCharacterTextSplitter` with overlap?**
BM25 retrieves at the chunk level. Without overlap, terms that span a chunk boundary are silently lost. A 200-character overlap guarantees that context crossing a split point appears in at least one chunk's token set, improving recall for multi-sentence answers.

**Why Ollama over an API-hosted LLM?**
Zero network dependency, no token cost, no data leaving the machine. The pipeline is designed to operate entirely on local hardware — this is a deliberate architectural constraint, not a workaround.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Managed via `.python-version` |
| [uv](https://docs.astral.sh/uv/) | Fast Python package and environment manager |
| [Ollama](https://ollama.com/download) | Local LLM runtime — must be running before `main.py` is executed |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/neo-bumblebee-ai/vectorless-rag.git
cd vectorless-rag
```

### 2. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2        # ~2 GB — recommended default
ollama pull llama3.2:1b     # ~800 MB — faster on lower-end hardware
ollama pull mistral         # alternative
```

### 4. Add your documents

```
data/
├── pdf_files/      ← drop any PDF here
└── text_files/     ← or plain .txt files here
```

A sample coffee machine text file (`sample_coffee_machine.txt`) is included so the demo works out of the box without any additional documents.

### 5. Run the pipeline

```bash
uv run python main.py
```

First run indexes everything automatically. Subsequent runs skip indexing entirely and go directly to the query loop.

---

## Bring Your Own Documents

1. **Remove the sample file** (optional):
   ```bash
   rm data/text_files/sample_coffee_machine.txt
   ```

2. **Add your documents:**
   ```
   data/pdf_files/     ← your PDFs
   data/text_files/    ← your .txt files
   ```

3. **Delete the existing index** to force re-indexing:
   ```bash
   rm -rf data/bm25_index/
   ```

4. **Run:**
   ```bash
   uv run python main.py
   ```

> The BM25 index is specific to the document set at build time. Always delete `data/bm25_index/` after adding or removing documents.

---

## Configuration

All settings are centralised at the top of `main.py`:

```python
OLLAMA_MODEL  = "llama3.2"          # any model from `ollama list`
PDF_DIR       = "data/pdf_files"    # where your PDFs live
TEXT_DIR      = "data/text_files"   # where your .txt files live
INDEX_DIR     = "data/bm25_index"   # where the BM25 index is persisted
CHUNK_SIZE    = 1000                # characters per chunk
CHUNK_OVERLAP = 200                 # overlap between adjacent chunks
TOP_K         = 5                   # chunks retrieved per query
```

### Swap the LLM

```bash
ollama pull mistral
```

```python
OLLAMA_MODEL = "mistral"
```

### Tune retrieval precision

```python
CHUNK_SIZE    = 500    # smaller chunks = more granular retrieval
CHUNK_OVERLAP = 100
TOP_K         = 8      # wider context window for the LLM
```

---

## Project Structure

```
vectorless-rag/
│
├── main.py                            # Full pipeline — entry point
│
├── notebooks/
│   └── bm25_walkthrough.ipynb         # Step-by-step: BM25 internals explained
│
├── data/
│   ├── pdf_files/                     # Input PDFs (not committed)
│   ├── text_files/
│   │   ├── sample_coffee_machine.txt  # Included demo document
│   │   └── YOUR_DOCUMENT.txt.example  # Rename and fill in for custom use
│   └── bm25_index/                    # Auto-generated index (not committed)
│       ├── index.pkl                  # Serialised BM25Okapi object
│       └── corpus.json                # Human-readable chunk corpus
│
├── .github/
│   ├── ISSUE_TEMPLATE/                # Bug report and feature request templates
│   └── pull_request_template.md
│
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT
├── pyproject.toml                     # Project metadata and dependencies
└── requirements.txt                   # pip-compatible dependency list
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama is not running. Start it with:
```bash
ollama serve
```

**`No documents found`**
Add files to `data/pdf_files/` or `data/text_files/` and re-run.

**Index appears stale after adding new documents**
Delete the index and re-run:
```bash
rm -rf data/bm25_index/
uv run python main.py
```

**Slow LLM responses**
- Switch to a smaller model: `ollama pull llama3.2:1b`
- Reduce `TOP_K` from 5 to 3 to limit context sent to the LLM

**Poor retrieval quality**
BM25 excels at exact keyword matching. If query terms differ significantly from document vocabulary:
- Rephrase using terminology that appears verbatim in the source documents
- Reduce `CHUNK_SIZE` to 500 for more granular retrieval
- Increase `TOP_K` to 8 to retrieve a broader candidate set

---

## Roadmap

- [ ] Hybrid retrieval — BM25 + dense embeddings with Reciprocal Rank Fusion (RRF)
- [ ] Re-ranking with a cross-encoder model
- [ ] Streamlit web UI
- [ ] Conversation memory for multi-turn follow-up questions
- [ ] RAGAS evaluation suite for retrieval and generation quality scoring
- [ ] Support for `.docx` and `.md` document formats

---

## Series

| Part | Project | Retrieval Approach |
|---|---|---|
| 2 | [Traditional RAG Pipeline](https://github.com/neo-bumblebee-ai/traditional-rag-pipeline) | Dense embeddings + ChromaDB |
| 3 | [Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph) | Self-correcting multi-agent retrieval loop |
| **4** | **Vectorless RAG** ← you are here | BM25 — no vectors, no embeddings |
| 5 | Coming May 2026 | — |

---

## Contributing

Contributions are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
