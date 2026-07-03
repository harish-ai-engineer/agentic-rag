# Agentic RAG over R2R — MCP + multi-agent handoffs + per-step cost + AgentGuard

Multi-agent retrieval system built on the locally running R2R instance
(`http://localhost:7272`).

```
User question
   │
   ▼
Triage Agent ──handoff──► Research Agent ──handoff──► Analyst Agent
                              │                            │
                    MCP client (Agents SDK)         local function tools
                              │                     (calculate, word_count)
                    MCP server: mcp_server.py
                              │
                    R2R API (search / rag / documents)
```

| Requirement | Where |
|---|---|
| Agentic retriever + relevance | `mcp_server.py::search_documents` — returns chunks with cosine-similarity scores, filters below `min_relevance` |
| Tools | MCP tools (`search_documents`, `rag_answer`, `list_documents`) + local function tools (`calculate`, `word_count`) |
| MCP server | `mcp_server.py` (FastMCP, stdio) |
| MCP client | `MCPServerStdio` in `agents_app.py` (OpenAI Agents SDK) |
| Multi-agent handoffs | Triage → Research → Analyst (`Agent(handoffs=[...])`) |
| Tool selection | LLM-driven; Research agent instructed when to prefer each tool |
| Per-step cost | `CostTracker` + `StepHooks` — one priced row per LLM call, tool call, handoff; table printed after each run |
| AgentGuard | OpenInference instrumentation → OTel → the AgentGuard OTel endpoint |

## Setup

Prereqs: R2R serving on `localhost:7272` (see repo docs), Python 3.10.

```powershell
cd Desktop\R2R\agentic-rag
py -3.10 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env    # then fill in keys (no quotes around values)
```

## Run

```powershell
.\.venv\Scripts\python.exe agents_app.py "What does Aristotle say about happiness?"
.\.venv\Scripts\python.exe agents_app.py "Summarize DeepSeek R1's training approach and count the words in your summary."
```

Each run prints the final answer, then a per-step cost table (tokens in/out
and USD per LLM step), and exports the full trace (agents, handoffs, tool
calls, generations) to AgentGuard if `AGENTGUARD_PUBLIC_KEY`/`AGENTGUARD_SECRET_KEY`
are set in `.env`.
