# Agentic RAG over ChromaDB

Multi-agent retrieval demo using a local ChromaDB vector store, MCP tools,
agent handoffs, per-step cost tracking, and AgentGuard traces.

The Chroma MCP server can sync document chunks from the local R2R API once.
After the sync, retrieval is served from the local ChromaDB collection.

```
User question
   |
   v
Triage Agent -> Research Agent -> Analyst Agent
                  |
                  | MCP client (Agents SDK)
                  v
            chroma_mcp_server.py
                  |
                  v
      Local ChromaDB persistent collection
```

| Requirement | Where |
|---|---|
| Vector database | ChromaDB persistent store at `CHROMA_DB_DIR` |
| Retriever + relevance | `chroma_mcp_server.py::search_documents` returns chunks with cosine relevance and filters below `min_relevance` |
| RAG answer | `chroma_mcp_server.py::rag_answer` retrieves from ChromaDB, then generates a cited answer |
| Document list | `chroma_mcp_server.py::list_documents` lists documents synced into ChromaDB |
| Sync | `chroma_mcp_server.py::sync_documents` copies chunks from local R2R into ChromaDB |
| MCP client | `MCPServerStdio` in `agents_app.py` starts the Chroma MCP server |
| Multi-agent handoffs | Triage -> Research -> Analyst |
| Local tools | `calculate`, `word_count` |
| Per-step cost | `CostTracker` + `StepHooks` prints one row per LLM call, tool call, and handoff |
| AgentGuard | OpenInference instrumentation exports OTel spans to AgentGuard |

## Setup

```powershell
cd C:\Users\SYS-02\Desktop\R2R\agentic-rag
py -3.10 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your keys. ChromaDB settings can stay as defaults:

```env
CHROMA_DB_DIR=./chroma_db
CHROMA_COLLECTION=agentic_rag_docs
CHROMA_AUTO_SYNC=true
EMBEDDING_MODEL=text-embedding-3-small
RAG_MODEL=gpt-4o-mini
AGENT_MODEL=gpt-4o-mini
```

## Run

```powershell
.\.venv\Scripts\python.exe agents_app.py "What does Aristotle say about happiness?"
```

On the first retrieval call, `chroma_mcp_server.py` syncs chunks into
`./chroma_db` if the Chroma collection is empty. Every later search reads from
ChromaDB.

Each run prints the final answer and a per-step cost table. If AgentGuard keys
are present in `.env`, the trace includes the agent workflow, handoffs, MCP tool
calls, token usage, and vector retrieval spans labeled `Vector DB: ChromaDB`.
