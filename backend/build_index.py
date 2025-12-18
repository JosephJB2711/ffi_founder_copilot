import os
import uuid
import hashlib
import requests
import chromadb
import pdfplumber
from docx import Document
from typing import Optional

# ─────────────────────────────
# CONFIG
# ─────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "ffi_founder_docs"

EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Chunking
MAX_CHARS = 1200
OVERLAP = 150

# Rebuild behaviour
# True = Collection wird gelöscht und komplett neu aufgebaut (empfohlen nach großen Änderungen)
REBUILD_COLLECTION = False


# ─────────────────────────────
# CHROMA INIT
# ─────────────────────────────

client = chromadb.PersistentClient(path=DB_DIR)

if REBUILD_COLLECTION:
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🧹 Collection '{COLLECTION_NAME}' gelöscht (REBUILD_COLLECTION=True).")
    except Exception:
        pass

collection = client.get_or_create_collection(COLLECTION_NAME)


# ─────────────────────────────
# HELPERS
# ─────────────────────────────

def classify_doc_type(filename: str) -> str:
    fn = filename.lower()
    if "satzung" in fn:
        return "satzung"
    if "datenschutz" in fn:
        return "datenschutz"
    if "event" in fn or "terms" in fn:
        return "event_terms"
    if "spons" in fn or "partner" in fn:
        return "sponsoring"
    return "other"


def stable_id(source: str, page: Optional[int], chunk_idx: int, text: str) -> str:
    """
    Deterministische ID, damit Re-Index ohne REBUILD nicht tausend Duplikate erzeugt.
    """
    key = f"{source}|{page or 0}|{chunk_idx}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """
    Absatzbasiertes Chunking + Overlap.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    cur = ""

    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p

    if cur:
        chunks.append(cur)

    if overlap and len(chunks) > 1:
        overlapped: list[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = chunks[i - 1]
                prefix = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append((prefix + "\n" + ch).strip())
        chunks = overlapped

    return chunks


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ─────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────

def extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())


def extract_pdf_pages(path: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            content = (page.extract_text() or "").strip()
            if content:
                pages.append((idx, content))
    return pages


def extract_plain_text_file(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith((".txt", ".md")):
        return extract_txt(path)
    if path_lower.endswith(".docx"):
        return extract_docx(path)
    raise ValueError(f"Unsupported file type for plain extraction: {path}")


# ─────────────────────────────
# INDEXING
# ─────────────────────────────

def index_documents():
    os.makedirs(DATA_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".txt", ".md", ".pdf", ".docx"))
    ]

    if not files:
        print("Keine unterstützten Dateien in ./data gefunden.")
        return

    total_added = 0

    for filename in files:
        full_path = os.path.join(DATA_DIR, filename)
        doc_type = classify_doc_type(filename)

        print(f"\n📄 Lese Datei: {filename} ({doc_type}) ...")

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        embeds: list[list[float]] = []

        if filename.lower().endswith(".pdf"):
            pages = extract_pdf_pages(full_path)
            if not pages:
                print(" → Keine extrahierbaren Textseiten gefunden (PDF evtl. gescannt).")
                continue

            chunks_all: list[tuple[int, int, str]] = []  # (page_no, chunk_idx, chunk_text)
            for page_no, page_text in pages:
                page_chunks = chunk_text(page_text)
                for chunk_idx, chunk in enumerate(page_chunks):
                    chunks_all.append((page_no, chunk_idx, chunk))

            print(f" → {len(chunks_all)} Chunks erzeugt (seitenbasiert).")

            for page_no, chunk_idx, chunk in chunks_all:
                try:
                    emb = get_embedding(chunk)
                except Exception as e:
                    print(f" ! Embedding-Fehler in {filename} p.{page_no} chunk {chunk_idx}: {e}")
                    continue

                _id = stable_id(filename, page_no, chunk_idx, chunk)

                ids.append(_id)
                docs.append(chunk)
                metas.append({
                    "source": filename,
                    "doc_type": doc_type,
                    "page": page_no,
                    "chunk": chunk_idx,
                })
                embeds.append(emb)

        else:
            text = extract_plain_text_file(full_path).strip()
            if not text:
                print(" → Datei ist leer oder nicht extrahierbar.")
                continue

            chunks = chunk_text(text)
            print(f" → {len(chunks)} Chunks erzeugt.")

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    emb = get_embedding(chunk)
                except Exception as e:
                    print(f" ! Embedding-Fehler in {filename} chunk {chunk_idx}: {e}")
                    continue

                _id = stable_id(filename, None, chunk_idx, chunk)

                ids.append(_id)
                docs.append(chunk)
                metas.append({
                    "source": filename,
                    "doc_type": doc_type,
                    "chunk": chunk_idx,
                })
                embeds.append(emb)

        if not ids:
            print(" → Nichts zum Hinzufügen.")
            continue

        # Hinweis: Chroma wirft bei Duplicate IDs i.d.R. Fehler.
        # Deshalb sind deterministische IDs hier Absicherung.
        try:
            collection.add(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
            total_added += len(ids)
            print(f" ✔ Indexiert: {filename} (+{len(ids)})")
        except Exception as e:
            print(f" ! Fehler beim Hinzufügen in Chroma ({filename}): {e}")

    print(f"\n🎉 Fertig! Insgesamt hinzugefügt: {total_added} Chunks.")
    print("CHROMA COUNT:", collection.count())


if __name__ == "__main__":
    index_documents()
