"""Stage 4-5: embeddings and the vector database."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CONFIG

COLLECTION = "papers"


def get_embeddings():
    CONFIG.validate()
    return GoogleGenerativeAIEmbeddings(
        model=CONFIG.embed_model,
        google_api_key=CONFIG.google_api_key,
    )


def get_store():
    CONFIG.validate()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CONFIG.persist_dir),
    )


def paper_exists(paper_id):
    store = get_store()
    try:
        got = store.get(where={"paper_id": paper_id}, limit=1)
        return bool(got.get("ids"))
    except Exception:
        return False


def add_documents(docs, batch_size=100):
    store = get_store()
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        ids = ["%s::%s" % (d.metadata["paper_id"], d.metadata["chunk_index"])
               for d in batch]
        store.add_documents(batch, ids=ids)
        total += len(batch)
    return total


def delete_paper(paper_id):
    get_store().delete(where={"paper_id": paper_id})


def list_papers():
    store = get_store()
    try:
        got = store.get(include=["metadatas"])
    except Exception:
        return []
    seen = {}
    for md in got.get("metadatas") or []:
        pid = md.get("paper_id")
        if not pid:
            continue
        if pid not in seen:
            seen[pid] = {
                "paper_id": pid,
                "title": md.get("title", pid),
                "source": md.get("source", ""),
                "chunks": 0,
            }
        seen[pid]["chunks"] += 1
    return sorted(seen.values(), key=lambda d: d["title"].lower())
