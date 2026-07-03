"""MCP server exposing the local R2R instance as retrieval tools.

Run standalone:  python mcp_server.py   (stdio transport)
The agent app launches this automatically via MCPServerStdio.
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

R2R_BASE_URL = os.getenv("R2R_BASE_URL", "http://localhost:7272")

mcp = FastMCP("r2r-retrieval")


@mcp.tool()
async def search_documents(
    query: str, top_k: int = 5, min_relevance: float = 0.2
) -> str:
    """Semantic search over the R2R knowledge base.

    Returns the top_k chunks with their relevance scores (0-1, cosine
    similarity). Chunks scoring below min_relevance are dropped so the
    caller only sees results worth citing.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{R2R_BASE_URL}/v3/retrieval/search",
            json={"query": query, "search_settings": {"limit": top_k}},
        )
        resp.raise_for_status()
        chunks = resp.json()["results"]["chunk_search_results"]

    kept = [c for c in chunks if c.get("score", 0) >= min_relevance]
    dropped = len(chunks) - len(kept)
    if not kept:
        return (
            f"No chunks above relevance {min_relevance} "
            f"({len(chunks)} candidates all scored lower)."
        )

    lines = []
    for i, c in enumerate(kept, 1):
        title = (c.get("metadata") or {}).get("title", "untitled")
        text = c.get("text", "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "..."
        lines.append(
            f"[{i}] relevance={c['score']:.3f} | doc='{title}' | "
            f"chunk_id={c['id'][:8]}\n{text}"
        )
    footer = f"\n({dropped} low-relevance chunks filtered out)" if dropped else ""
    return "\n\n".join(lines) + footer


@mcp.tool()
async def rag_answer(query: str) -> str:
    """Ask R2R's built-in RAG pipeline for a fully-formed, citation-backed
    answer. Use for direct factual questions; use search_documents when you
    need raw evidence chunks to reason over yourself.
    """
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{R2R_BASE_URL}/v3/retrieval/rag", json={"query": query}
        )
        resp.raise_for_status()
        results = resp.json()["results"]
    answer = results.get("generated_answer", "")
    cites = {
        (c.get("payload") or {}).get("metadata", {}).get("title")
        for c in results.get("citations", [])
        if isinstance(c, dict)
    }
    cites.discard(None)
    if cites:
        answer += "\n\nSources: " + ", ".join(sorted(cites))
    return answer


@mcp.tool()
async def list_documents() -> str:
    """List all documents currently in the R2R knowledge base with their
    ingestion status and summaries."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{R2R_BASE_URL}/v3/documents")
        resp.raise_for_status()
        docs = resp.json()["results"]
    if not docs:
        return "Knowledge base is empty."
    lines = []
    for d in docs:
        summary = (d.get("summary") or "")[:200]
        lines.append(
            f"- {d.get('title', 'untitled')} [{d['document_type']}] "
            f"status={d['ingestion_status']} tokens={d.get('total_tokens')}\n"
            f"  {summary}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
