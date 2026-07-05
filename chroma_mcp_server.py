"""MCP server exposing a local ChromaDB store as retrieval tools.

The first call auto-syncs chunks from the local R2R server into Chroma when
the collection is empty. After that, retrieval is served from ChromaDB.
"""

import os
import sys
from pathlib import Path
from typing import Any

import chromadb
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR / ".env")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R2R_BASE_URL = os.getenv("R2R_BASE_URL", "http://localhost:7272")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(PROJECT_DIR / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "agentic_rag_docs")
CHROMA_AUTO_SYNC = os.getenv("CHROMA_AUTO_SYNC", "true").lower() != "false"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_MODEL = os.getenv("RAG_MODEL", os.getenv("AGENT_MODEL", "gpt-4o-mini"))

mcp = FastMCP("chroma-retrieval")
openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


def _embed(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


async def _fetch_r2r_docs_and_chunks() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=120) as client:
        docs_resp = await client.get(f"{R2R_BASE_URL}/v3/documents")
        docs_resp.raise_for_status()
        docs = docs_resp.json()["results"]

        chunks: list[dict[str, Any]] = []
        for doc in docs:
            offset = 0
            while True:
                chunk_resp = await client.get(
                    f"{R2R_BASE_URL}/v3/documents/{doc['id']}/chunks",
                    params={"offset": offset, "limit": 100},
                )
                chunk_resp.raise_for_status()
                payload = chunk_resp.json()
                results = payload.get("results", [])
                if not results:
                    break
                for chunk in results:
                    chunk_metadata = chunk.get("metadata") or {}
                    chunks.append(
                        {
                            "id": chunk["id"],
                            "text": chunk.get("text", ""),
                            "metadata": _clean_metadata(
                                {
                                    **chunk_metadata,
                                    "chunk_id": chunk["id"],
                                    "document_id": chunk["document_id"],
                                    "doc_title": doc.get("title", "untitled"),
                                    "document_type": doc.get(
                                        "document_type",
                                        chunk_metadata.get("document_type", ""),
                                    ),
                                    "ingestion_status": doc.get(
                                        "ingestion_status", ""
                                    ),
                                }
                            ),
                        }
                    )
                offset += len(results)
                total = payload.get("total_entries")
                if total is not None and offset >= total:
                    break
        return chunks


async def _sync_from_r2r(reset: bool = False) -> str:
    if reset:
        existing = collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)

    chunks = await _fetch_r2r_docs_and_chunks()
    if not chunks:
        return "No R2R chunks found to sync."

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = _embed(texts)
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )

    doc_titles = {
        item["metadata"].get("doc_title", "untitled") for item in chunks
    }
    return (
        f"Synced {len(chunks)} chunks from {len(doc_titles)} documents "
        f"into ChromaDB collection '{CHROMA_COLLECTION}'."
    )


async def _ensure_synced() -> None:
    if collection.count() == 0 and CHROMA_AUTO_SYNC:
        await _sync_from_r2r(reset=False)


def _search_chroma(
    query: str,
    top_k: int,
    min_relevance: float,
) -> list[dict[str, Any]]:
    query_embedding = _embed([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for item_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        relevance = max(0.0, min(1.0, 1.0 - float(distance)))
        if relevance < min_relevance:
            continue
        rows.append(
            {
                "id": item_id,
                "text": text or "",
                "metadata": metadata or {},
                "relevance": relevance,
                "distance": float(distance),
            }
        )
    return rows


@mcp.tool()
async def sync_documents(reset: bool = False) -> str:
    """Sync R2R document chunks into the local ChromaDB collection."""
    return await _sync_from_r2r(reset=reset)


@mcp.tool()
async def search_documents(
    query: str, top_k: int = 5, min_relevance: float = 0.2
) -> str:
    """Semantic search over the local ChromaDB knowledge base."""
    await _ensure_synced()
    rows = _search_chroma(query, top_k, min_relevance)
    if not rows:
        return (
            f"No ChromaDB chunks above relevance {min_relevance}. "
            f"Collection count={collection.count()}."
        )

    lines = []
    for i, row in enumerate(rows, 1):
        metadata = row["metadata"]
        title = metadata.get("doc_title", "untitled")
        text = row["text"].strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "..."
        lines.append(
            f"[{i}] relevance={row['relevance']:.3f} | "
            f"vector_db=ChromaDB | doc='{title}' | "
            f"chunk_id={str(row['id'])[:8]}\n{text}"
        )
    return "\n\n".join(lines)


@mcp.tool()
async def rag_answer(query: str) -> str:
    """Answer using ChromaDB retrieval plus an OpenAI generation step."""
    await _ensure_synced()
    rows = _search_chroma(query, top_k=6, min_relevance=0.2)
    if not rows:
        return "No relevant ChromaDB chunks found."

    context_blocks = []
    for i, row in enumerate(rows, 1):
        metadata = row["metadata"]
        context_blocks.append(
            f"[{i}] title={metadata.get('doc_title', 'untitled')} "
            f"relevance={row['relevance']:.3f} "
            f"chunk_id={str(row['id'])[:8]}\n{row['text']}"
        )

    response = openai_client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied ChromaDB context. "
                    "Cite source numbers and relevance scores."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    "ChromaDB context:\n\n" + "\n\n".join(context_blocks)
                ),
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    sources = sorted(
        {
            str(row["metadata"].get("doc_title", "untitled"))
            for row in rows
        }
    )
    return answer + "\n\nVector DB: ChromaDB\nSources: " + ", ".join(sources)


@mcp.tool()
async def list_documents() -> str:
    """List documents currently synced into the local ChromaDB collection."""
    await _ensure_synced()
    count = collection.count()
    if count == 0:
        return "ChromaDB collection is empty."

    payload = collection.get(include=["metadatas"], limit=count)
    docs: dict[str, dict[str, Any]] = {}
    for metadata in payload.get("metadatas", []):
        if not metadata:
            continue
        doc_id = str(metadata.get("document_id", "unknown"))
        docs.setdefault(
            doc_id,
            {
                "title": metadata.get("doc_title", "untitled"),
                "document_type": metadata.get("document_type", ""),
                "chunks": 0,
            },
        )
        docs[doc_id]["chunks"] += 1

    lines = [
        f"ChromaDB collection '{CHROMA_COLLECTION}' has {count} chunks."
    ]
    for item in sorted(docs.values(), key=lambda value: str(value["title"])):
        lines.append(
            f"- {item['title']} [{item['document_type']}] "
            f"chunks={item['chunks']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
