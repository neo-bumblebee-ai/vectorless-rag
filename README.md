# Vectorless RAG Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com)
[![Retrieval](https://img.shields.io/badge/retrieval-BM25-brightgreen.svg)](https://github.com/dorianbrown/rank_bm25)
[![No Vectors](https://img.shields.io/badge/vectors-none-red.svg)](#why-no-vectors)

> **Part 4 of a hands-on AI engineering series** — building in public, month by month.
>
> ← [Part 3 — Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph) &nbsp;|&nbsp; Part 5 — coming May 2026

---

## What Is This?

Every other RAG tutorial starts the same way: generate embeddings, spin up a vector database, store 384-dimensional floats, query by cosine similarity. It works — but it's heavy.

This project strips all of that out.

**Vectorless RAG** replaces the embedding + vector-store layer with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) — the same battle-tested ranking algorithm that powers search engines. No GPU, no embedding model, no external database. Just fast keyword-aware retrieval that runs anywhere Python runs.

> Drop in your own PDFs or text files, ask questions in plain English, and get grounded answers — no vector store required.

---

## Demo

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
| Index size | Large (float arrays) | Small (inverted index) |
| First-run time | Minutes (embedding batch) | **Seconds** |
| GPU / RAM pressure | High | **Minimal** |
| Exact keyword match | Weak | **Strong** |
| Semantic similarity | Strong | Weaker |
| Works offline | Yes (with local models) | **Yes** |

**When to use BM25 over vectors:**
- Your documents use precise terminology (manuals, legal docs, medical records)
- You want instant startup — no re-embedding on every restart
- You're running on constrained hardware
- You want to understand RAG fundamentals without framework magic

---

## Architecture

```
Your Documents (PDFs / .txt)
        │
        ▼
┌───────────────────┐
│   Document Loader │  LangChain PyPDFLoader / TextLoader
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Text Splitter   │  RecursiveCharacterTextSplitter
│   chunk=1000      │  overlap=200 chars
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   BM25 Indexer    │  rank-bm25 / BM25Okapi
│   (no embeddings) │  tokenise → inverted index
└────────┬──────────┘
         │  persisted to data/bm25_index/ as .pkl + .json
         ▼
┌───────────────────┐
│   BM25 Retriever  │  keyword scoring → top-K chunks
│   (at query time) │  loads from disk instantly on restart
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Ollama LLM      │  llama3.2 (or any model you pull)
│   (runs locally)  │  context-grounded answer generation
└───────────────────┘
```

**On first run:** documents are loaded, chunked, and indexed with BM25. The index is saved to disk.
**On subsequent runs:** the index is loaded from disk in milliseconds — no re-processing.

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Document loading | LangChain | `PyPDFLoader`, `TextLoader` |
| Text splitting | `RecursiveCharacterTextSplitter` | Configurable chunk size & overlap |
| Retrieval | `rank-bm25` (`BM25Okapi`) | No embeddings, no GPU |
| Index persistence | `pickle` + `json` | Instant reload on restart |
| LLM | Ollama | `llama3.2` by default, fully local |
| Package manager | `uv` | Fast Python package manager |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Ollama](https://ollama.com/download) — local LLM runner

---

## Quick Start

### 1. Clone the repo

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
ollama pull llama3.2        # ~2 GB — recommended
ollama pull llama3.2:1b     # ~800 MB — faster on low-end hardware
ollama pull mistral         # alternative
```

### 4. Add your documents

```
data/
├── pdf_files/      ← drop any PDF here
└── text_files/     ← or plain .txt files here
```

A sample coffee machine text file (`sample_coffee_machine.txt`) is included so the demo works out of the box.

### 5. Run

```bash
uv run python main.py
```

First run indexes everything automatically. Subsequent runs skip straight to the query loop.

---

## Bring Your Own Documents

This repo is designed to be swapped out. To use your own documents:

1. **Remove the sample file** (optional):
   ```bash
   rm data/text_files/sample_coffee_machine.txt
   ```

2. **Add your documents**:
   ```
   data/pdf_files/     ← your PDFs
   data/text_files/    ← your .txt files
   ```

3. **Delete the existing index** (forces re-indexing):
   ```bash
   rm -rf data/bm25_index/
   ```

4. **Run**:
   ```bash
   uv run python main.py
   ```

> **Tip:** The index is specific to your document set. Always delete `data/bm25_index/` after adding or removing documents.

---

## Configuration

All settings live at the top of `main.py`:

```python
OLLAMA_MODEL  = "llama3.2"      # ← any model from `ollama list`
PDF_DIR       = "data/pdf_files"   # ← where your PDFs live
TEXT_DIR      = "data/text_files"  # ← where your .txt files live
INDEX_DIR     = "data/bm25_index"  # ← where the BM25 index is saved
CHUNK_SIZE    = 1000               # ← characters per chunk
CHUNK_OVERLAP = 200                # ← overlap between chunks
TOP_K         = 5                  # ← chunks retrieved per query
```

### Swap the LLM

```bash
ollama pull mistral
```

```python
OLLAMA_MODEL = "mistral"
```

### Tune retrieval

```python
CHUNK_SIZE    = 500   # smaller chunks = more precise retrieval
CHUNK_OVERLAP = 100
TOP_K         = 8     # more context sent to the LLM
```

### Use a different document directory

```python
PDF_DIR  = "/path/to/your/pdfs"
TEXT_DIR = "/path/to/your/txts"
```

---

## Project Structure

```
vectorless-rag/
│
├── main.py                         # Full pipeline — run this
│
├── notebooks/
│   └── bm25_walkthrough.ipynb      # Step-by-step: BM25 internals explained
│
├── data/
│   ├── pdf_files/                  # Your PDFs go here (not committed)
│   │   └── .gitkeep
│   ├── text_files/                 # Sample + your .txt files
│   │   ├── sample_coffee_machine.txt   ← working demo
│   │   └── YOUR_DOCUMENT.txt.example  ← rename & fill in
│   └── bm25_index/                 # BM25 index — auto-generated (not committed)
│
├── .github/
│   ├── ISSUE_TEMPLATE/             # Bug report & feature request templates
│   └── pull_request_template.md
│
├── CONTRIBUTING.md                 # How to contribute
├── LICENSE                         # MIT
├── pyproject.toml                  # Project metadata & dependencies
└── requirements.txt                # pip-compatible dependency list
```

---

## Troubleshooting

**`Failed to connect to Ollama`**
Ollama isn't running. Start it:
```bash
ollama serve
```

**`No documents found`**
Add files to `data/pdf_files/` or `data/text_files/` and re-run.

**Index seems stale after adding new documents**
Delete the index and re-run:
```bash
rm -rf data/bm25_index/
uv run python main.py
```

**Slow responses**
- Switch to a smaller model: `ollama pull llama3.2:1b`
- Reduce `TOP_K` from 5 to 3 to send less context

**Poor retrieval quality**
BM25 excels at keyword matching. If your questions use different terminology from your documents, try:
- Rephrasing with exact terms from your docs
- Reducing `CHUNK_SIZE` to 500 for more granular chunks
- Increasing `TOP_K` to 8 to cast a wider net

---

## Roadmap

- [ ] Hybrid retrieval (BM25 + dense embeddings with RRF fusion)
- [ ] Re-ranking with a cross-encoder
- [ ] Streamlit web UI
- [ ] Conversation memory for follow-up questions
- [ ] RAGAS evaluation suite
- [ ] Support for `.docx` and `.md` documents

---

## Series

| Part | Project | Focus |
|---|---|---|
| 2 | [Traditional RAG Pipeline](https://github.com/neo-bumblebee-ai/traditional-rag-pipeline) | Embeddings + ChromaDB |
| 3 | [Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph) | Self-correcting multi-agent loop |
| **4** | **Vectorless RAG** ← you are here | BM25 — no vectors, no embeddings |
| 5 | Coming May 2026 | — |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
