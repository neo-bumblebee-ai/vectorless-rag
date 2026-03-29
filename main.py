"""
Vectorless RAG Pipeline
Ingest → Chunk → Index → Retrieve → Generate

Documents : PDF and text files under data/
Retrieval : BM25 (rank-bm25) — no embeddings, no vector database
Index     : Persisted to data/bm25_index/ — skips re-indexing on restart
Generation: Ollama (local LLM)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ── Configuration ─────────────────────────────────────────────────────────────
# ↓↓ EDIT THESE to match your setup ↓↓

OLLAMA_MODEL  = "llama3.2"          # any model from `ollama list`
                                    # alternatives: "mistral", "llama3.2:1b", "phi3"

PDF_DIR       = "data/pdf_files"    # drop your PDFs here
TEXT_DIR      = "data/text_files"   # or plain .txt files here
INDEX_DIR     = "data/bm25_index"   # persisted BM25 index lives here

CHUNK_SIZE    = 1000                # characters per chunk — lower = more granular
CHUNK_OVERLAP = 200                 # overlap between adjacent chunks
TOP_K         = 5                   # chunks retrieved per query

# ↑↑ Nothing below this line needs to change for basic use ↑↑
# ─────────────────────────────────────────────────────────────────────────────

INDEX_FILE  = Path(INDEX_DIR) / "index.pkl"
CORPUS_FILE = Path(INDEX_DIR) / "corpus.json"


# ── Document loading ──────────────────────────────────────────────────────────
def load_documents() -> list[Document]:
    docs: list[Document] = []

    pdf_path = Path(PDF_DIR)
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        print(f"  Loading PDF : {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page in pages:
            page.metadata.update({"source_file": pdf_file.name, "file_type": "pdf"})
        docs.extend(pages)

    text_path = Path(TEXT_DIR)
    for txt_file in sorted(text_path.glob("*.txt")):
        print(f"  Loading TXT : {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        pages = loader.load()
        for page in pages:
            page.metadata.update({"source_file": txt_file.name, "file_type": "text"})
        docs.extend(pages)

    if not docs:
        print(
            "\n[!] No documents found.\n"
            f"    • Drop PDFs into  : {PDF_DIR}/\n"
            f"    • Drop .txt files : {TEXT_DIR}/\n"
            "    Then re-run.\n"
        )
    else:
        print(f"Loaded {len(docs)} document pages total.")

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ── BM25 tokeniser ────────────────────────────────────────────────────────────
def _tokenise(text: str) -> list[str]:
    """Lowercase, split on whitespace — fast and dependency-free."""
    return text.lower().split()


# ── Index management ──────────────────────────────────────────────────────────
class BM25Index:
    """BM25 retriever with on-disk persistence (no re-indexing on restart)."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.corpus: list[dict[str, Any]] = []   # [{text, metadata}, …]

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self, chunks: list[Document]) -> None:
        print(f"Building BM25 index over {len(chunks)} chunks …")
        self.corpus = [
            {"text": c.page_content, "metadata": c.metadata} for c in chunks
        ]
        tokenised = [_tokenise(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenised)
        print("Index built.")

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self) -> None:
        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False)
        print(f"Index saved to {INDEX_DIR}/")

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Return True if a saved index was loaded, False otherwise."""
        if not (INDEX_FILE.exists() and CORPUS_FILE.exists()):
            return False
        with open(INDEX_FILE, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(CORPUS_FILE, encoding="utf-8") as f:
            self.corpus = json.load(f)
        print(
            f"BM25 index loaded from disk — {len(self.corpus)} chunks "
            f"(skipping re-indexing)."
        )
        return True

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if self.bm25 is None:
            raise RuntimeError("Index is not built or loaded yet.")
        tokens = _tokenise(question)
        scores = self.bm25.get_scores(tokens)

        # Pair every chunk with its BM25 score, sort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            {
                "document": self.corpus[i]["text"],
                "metadata": self.corpus[i]["metadata"],
                "score": float(score),
            }
            for i, score in ranked
        ]


# ── Generation ────────────────────────────────────────────────────────────────
def generate_answer(
    question: str,
    context_docs: list[dict[str, Any]],
    model: str = OLLAMA_MODEL,
) -> str:
    context_parts = []
    for d in context_docs:
        source = d["metadata"].get("source_file", "unknown")
        page   = d["metadata"].get("page", "?")
        context_parts.append(f"[Source: {source}, page {page}]\n{d['document']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the question using only the context below.\n"
        "If the answer is not in the context, say: "
        '"I don\'t have enough information to answer that."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ── Pipeline helpers ──────────────────────────────────────────────────────────
def ingest(index: BM25Index) -> None:
    """Load, chunk, and index all documents. Skips if already indexed."""
    if index.load():
        print()
        return

    print("\n── Ingesting documents ──────────────────────────────────────────────")
    documents = load_documents()
    if not documents:
        return

    chunks = split_documents(documents)
    index.build(chunks)
    index.save()
    print("Ingestion complete.\n")


def rag_query(question: str, index: BM25Index) -> str:
    """Retrieve context via BM25 and generate an answer."""
    results = index.query(question, top_k=TOP_K)

    print(f"\nTop {TOP_K} retrieved chunks:")
    for i, r in enumerate(results, 1):
        src     = r["metadata"].get("source_file", "?")
        pg      = r["metadata"].get("page", "?")
        snippet = r["document"][:80].replace("\n", " ").strip()
        print(f"  {i}. [score {r['score']:.2f}] {src} p.{pg} — {snippet}…")

    print("\nGenerating answer with Ollama …")
    return generate_answer(question, results)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Vectorless RAG Pipeline ===\n")

    index = BM25Index()
    ingest(index)

    if not index.corpus:
        print("No documents indexed. Add files to data/ and re-run.")
        return

    print(f"Ollama model : {OLLAMA_MODEL}")
    print("Type a question or 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        answer = rag_query(question, index)
        print(f"\nAnswer:\n{answer}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()
