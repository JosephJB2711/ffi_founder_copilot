import os
import chromadb

# ─────────────────────────────
# Pfade exakt wie in main.py / build_index.py
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "ffi_founder_docs"

print("BASE_DIR:", BASE_DIR)
print("DB_DIR:", DB_DIR)
print("COLLECTION:", COLLECTION_NAME)
print("-" * 60)

# ─────────────────────────────
# Chroma öffnen
# ─────────────────────────────
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

print("TOTAL CHUNKS IN COLLECTION:", collection.count())
print("-" * 60)

# ─────────────────────────────
# 1) Stichprobe: Gibt es doc_type überhaupt?
# ─────────────────────────────
peek = collection.peek(5)
metas = peek.get("metadatas", [])

print("PEEK METADATA SAMPLE:")
for i, m in enumerate(metas):
    print(f"{i+1}.", m)

print("-" * 60)

# ─────────────────────────────
# 2) Existiert doc_type = 'satzung'?
# ─────────────────────────────
try:
    res = collection.get(
        where={"doc_type": "satzung"},
        limit=5,
        include=["metadatas"]
    )
    metas = res.get("metadatas", [])
    print("SATZUNG CHUNKS FOUND:", len(metas))
    if metas:
        print("FIRST SATZUNG META:", metas[0])
except Exception as e:
    print("ERROR querying doc_type='satzung':", e)

print("-" * 60)

# ─────────────────────────────
# 3) Welche doc_types gibt es überhaupt?
# ─────────────────────────────
print("COLLECTING DISTINCT doc_type VALUES (sampled)…")

doc_types = set()
sample = collection.peek(50)
for m in sample.get("metadatas", []):
    if m and "doc_type" in m:
        doc_types.add(m["doc_type"])

print("FOUND doc_type VALUES:", doc_types)
print("-" * 60)

print("INSPECTION DONE.")
