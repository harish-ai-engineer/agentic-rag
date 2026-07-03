"""Multi-agent RAG over ChromaDB with MCP tools, handoffs, per-step cost, and
Langfuse observability.

Architecture
------------
  Triage Agent ──handoff──> Research Agent ──handoff──> Analyst Agent
       │                        │  MCP tools (chroma-retrieval server):
       │                        │    search_documents (+relevance filter)
       │                        │    rag_answer, list_documents, sync_documents
       │                        └─ tool selection is LLM-driven
       └─ routes the user request to the right specialist

  Per-step cost: RunHooks capture every LLM call's token usage and price it.
  Observability: OpenInference instrumentation exports OTel spans to
  AgentGuard (Langfuse-compatible API).

Usage
-----
  python agents_app.py "What does Aristotle say about happiness?"
  python agents_app.py            # runs the default demo question
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROJECT_DIR = Path(__file__).parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# AgentGuard observability via OpenTelemetry (no-op if keys are missing)
# ---------------------------------------------------------------------------

DEFAULT_AGENTGUARD_HOST = (
    "https://agentgaurd-a0acc6egbhced0dc.centralindia-01.azurewebsites.net"
)


def setup_agentguard() -> bool:
    pk = os.getenv("AGENTGUARD_PUBLIC_KEY")
    sk = os.getenv("AGENTGUARD_SECRET_KEY")
    host = os.getenv("AGENTGUARD_HOST", DEFAULT_AGENTGUARD_HOST).rstrip("/")
    if not (pk and sk):
        return False

    from openinference.instrumentation.openai_agents import (
        OpenAIAgentsInstrumentor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    provider = TracerProvider(
        resource=Resource.create({"service.name": "r2r-agentic-rag"})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    globals()["_otel_provider"] = provider
    globals()["_otel_tracer"] = provider.get_tracer("r2r-agentic-rag")
    return True


# ---------------------------------------------------------------------------
# Per-step cost tracking
# ---------------------------------------------------------------------------

# USD per 1M tokens (input, output)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-2025-08-07": (1.25, 10.00),
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
}


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    for key, (pin, pout) in PRICING.items():
        if model.startswith(key):
            return tokens_in / 1e6 * pin + tokens_out / 1e6 * pout
    return 0.0


class CostTracker:
    """Collects one row per LLM call / tool call / handoff."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add_llm(self, agent_name: str, model: str, usage) -> None:
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        self.steps.append(
            {
                "kind": "llm",
                "agent": agent_name,
                "detail": model,
                "in": tokens_in,
                "out": tokens_out,
                "usd": price(model, tokens_in, tokens_out),
            }
        )

    def add_event(self, kind: str, agent_name: str, detail: str) -> None:
        self.steps.append(
            {"kind": kind, "agent": agent_name, "detail": detail,
             "in": 0, "out": 0, "usd": 0.0}
        )

    @property
    def total_usd(self) -> float:
        return sum(s["usd"] for s in self.steps)

    def report(self) -> str:
        lines = [
            "",
            "=" * 78,
            "PER-STEP COST REPORT",
            "=" * 78,
            f"{'#':>2}  {'kind':<8} {'agent':<16} {'detail':<30} "
            f"{'in':>6} {'out':>6} {'$':>9}",
            "-" * 78,
        ]
        for i, s in enumerate(self.steps, 1):
            lines.append(
                f"{i:>2}  {s['kind']:<8} {s['agent']:<16} "
                f"{s['detail'][:30]:<30} {s['in']:>6} {s['out']:>6} "
                f"{s['usd']:>9.6f}"
            )
        lines.append("-" * 78)
        lines.append(f"{'TOTAL':>62} {self.total_usd:>15.6f} USD")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

from agents import Agent, RunHooks, Runner, function_tool  # noqa: E402
from agents.mcp import MCPServerStdio  # noqa: E402

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")


class StepHooks(RunHooks):
    def __init__(self, tracker: CostTracker, user_question: str) -> None:
        self.tracker = tracker
        self.user_question = user_question
        self.tool_spans: dict[str, list[object]] = {}
        self.llm_spans: dict[str, list[object]] = {}

    @staticmethod
    def _tool_key(context, tool) -> str:
        call_id = str(getattr(context, "tool_call_id", "") or "")
        return call_id or getattr(tool, "name", "tool")

    @staticmethod
    def _safe_json(value: object, limit: int = 12000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        return text[:limit]

    @staticmethod
    def _set_current_span_io(
        input_value: str | None = None,
        output_value: str | None = None,
        mime_type: str = "application/json",
    ) -> None:
        try:
            from opentelemetry import trace
            from openinference.semconv.trace import SpanAttributes

            span = trace.get_current_span()
            if input_value is not None:
                span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime_type)
                span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
            if output_value is not None:
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime_type)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
        except Exception:
            return

    async def on_agent_start(self, context, agent) -> None:
        payload = {
            "agent": agent.name,
            "user_question": self.user_question,
            "turn_input": getattr(context, "turn_input", None),
        }
        input_value = self._safe_json(payload)
        self._set_current_span_io(input_value=input_value)

        tracer = globals().get("_otel_tracer")
        if tracer is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        with tracer.start_as_current_span(f"Agent input: {agent.name}") as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.AGENT.value,
            )
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE,
                               OpenInferenceMimeTypeValues.JSON.value)
            span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)

    async def on_agent_end(self, context, agent, output) -> None:
        output_value = str(output)[:12000]
        self._set_current_span_io(output_value=output_value, mime_type="text/plain")

        tracer = globals().get("_otel_tracer")
        if tracer is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        with tracer.start_as_current_span(f"Agent output: {agent.name}") as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.AGENT.value,
            )
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE,
                               OpenInferenceMimeTypeValues.TEXT.value)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)

    async def on_llm_start(
        self, context, agent, system_prompt, input_items
    ) -> None:
        payload = {
            "agent": agent.name,
            "user_question": self.user_question,
            "system_prompt": system_prompt,
            "input_items": input_items,
        }
        input_value = self._safe_json(payload)
        self._set_current_span_io(input_value=input_value)

        tracer = globals().get("_otel_tracer")
        if tracer is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        span = tracer.start_span(f"Agent turn input: {agent.name}")
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE,
                           OpenInferenceMimeTypeValues.JSON.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
        self.llm_spans.setdefault(agent.name, []).append(span)

    async def on_llm_end(self, context, agent, response) -> None:
        model = getattr(agent, "model", None) or MODEL
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.tracker.add_llm(agent.name, str(model), usage)
        output_value = self._safe_json(getattr(response, "output", response))
        self._set_current_span_io(output_value=output_value)

        spans = self.llm_spans.get(agent.name, [])
        span = spans.pop() if spans else None
        if not spans:
            self.llm_spans.pop(agent.name, None)
        if span is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            SpanAttributes,
        )

        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE,
                           OpenInferenceMimeTypeValues.JSON.value)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
        span.end()

    async def on_tool_start(self, context, agent, tool) -> None:
        self.tracker.add_event("tool", agent.name, tool.name)
        if tool.name not in {
            "search_documents",
            "rag_answer",
            "list_documents",
            "sync_documents",
        }:
            return

        tracer = globals().get("_otel_tracer")
        if tracer is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        tool_args = getattr(context, "tool_arguments", None)
        span = tracer.start_span(f"Vector DB: ChromaDB / {tool.name}")
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.RETRIEVER.value,
        )
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE,
                           OpenInferenceMimeTypeValues.JSON.value)
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            json.dumps(
                {
                    "agent": agent.name,
                    "tool": tool.name,
                    "arguments": tool_args,
                    "chroma_db_dir": os.getenv(
                        "CHROMA_DB_DIR", str(PROJECT_DIR / "chroma_db")
                    ),
                    "chroma_collection": os.getenv(
                        "CHROMA_COLLECTION", "agentic_rag_docs"
                    ),
                    "database": "ChromaDB",
                    "storage": "Local ChromaDB persistent collection",
                },
                ensure_ascii=True,
            ),
        )
        self.tool_spans.setdefault(self._tool_key(context, tool), []).append(span)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self.tracker.add_event(
            "handoff", from_agent.name, f"-> {to_agent.name}"
        )
        self._set_current_span_io(
            input_value=self._safe_json(
                {
                    "from_agent": from_agent.name,
                    "to_agent": to_agent.name,
                    "user_question": self.user_question,
                    "turn_input": getattr(context, "turn_input", None),
                }
            )
        )

    async def on_tool_end(self, context, agent, tool, result: object) -> None:
        key = self._tool_key(context, tool)
        spans = self.tool_spans.get(key) or self.tool_spans.get(tool.name) or []
        span = spans.pop() if spans else None
        if not spans:
            self.tool_spans.pop(key, None)
            self.tool_spans.pop(tool.name, None)
        if span is None:
            return

        from openinference.semconv.trace import (
            OpenInferenceMimeTypeValues,
            SpanAttributes,
        )

        output = str(result)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE,
                           OpenInferenceMimeTypeValues.TEXT.value)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output[:12000])
        span.set_attribute("chroma.result_preview", output[:1200])
        span.end()

    def close_open_tool_spans(self) -> None:
        for spans in self.tool_spans.values():
            while spans:
                span = spans.pop()
                span.set_attribute("chroma.warning", "tool output was not captured")
                span.end()
        self.tool_spans.clear()
        for spans in self.llm_spans.values():
            while spans:
                span = spans.pop()
                span.set_attribute("agent.warning", "LLM output was not captured")
                span.end()
        self.llm_spans.clear()


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '0.15 * 1234 / 1e6'."""
    allowed = set("0123456789.+-*/()e %")
    if not set(expression) <= allowed:
        return "Error: only arithmetic characters are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@function_tool
def word_count(text: str) -> str:
    """Count words in a text."""
    return str(len(text.split()))


def build_agents(chroma_mcp: MCPServerStdio):
    analyst = Agent(
        name="Analyst",
        model=MODEL,
        instructions=(
            "You are the Analyst. You receive research findings and compose "
            "the final answer for the user: concise, structured, and citing "
            "document titles and relevance scores where provided. Use "
            "calculate/word_count when numbers are involved. You are the "
            "last step - do NOT hand off."
        ),
        tools=[calculate, word_count],
    )

    research = Agent(
        name="Research",
        model=MODEL,
        instructions=(
            "You are the Research agent for a ChromaDB knowledge base. "
            "Pick tools deliberately: list_documents to see what exists; "
            "search_documents for raw evidence chunks with relevance scores "
            "(cite them, ignore results below 0.2 relevance); rag_answer "
            "for a direct citation-backed answer. Gather evidence, then "
            "always hand off to the Analyst with a bullet summary of findings "
            "including relevance scores. Do not write the final answer yourself."
        ),
        mcp_servers=[chroma_mcp],
        handoffs=[analyst],
    )

    triage = Agent(
        name="Triage",
        model=MODEL,
        instructions=(
            "You are the Triage agent. Route the user's request: anything "
            "needing knowledge-base lookup goes to Research; pure "
            "writing/calculation tasks on provided text go straight to "
            "Analyst. Hand off immediately - do not answer yourself."
        ),
        handoffs=[research, analyst],
    )
    return triage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_QUESTION = (
    "What does Aristotle say about happiness, and how many documents are in "
    "the knowledge base? End with a one-line take-away."
)


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    ag_on = setup_agentguard()
    print(f"AgentGuard export: {'ON' if ag_on else 'OFF (no keys in .env)'}")
    print(f"Question: {question}\n")

    tracker = CostTracker()
    hooks = StepHooks(tracker, question)
    chroma_mcp = MCPServerStdio(
        name="chroma-retrieval",
        params={
            "command": sys.executable,
            "args": [str(PROJECT_DIR / "chroma_mcp_server.py")],
            "env": {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "R2R_BASE_URL": os.getenv(
                    "R2R_BASE_URL", "http://localhost:7272"
                ),
                "CHROMA_DB_DIR": os.getenv(
                    "CHROMA_DB_DIR", str(PROJECT_DIR / "chroma_db")
                ),
                "CHROMA_COLLECTION": os.getenv(
                    "CHROMA_COLLECTION", "agentic_rag_docs"
                ),
                "CHROMA_AUTO_SYNC": os.getenv("CHROMA_AUTO_SYNC", "true"),
                "EMBEDDING_MODEL": os.getenv(
                    "EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                "RAG_MODEL": os.getenv("RAG_MODEL", MODEL),
                "AGENT_MODEL": MODEL,
            },
        },
        client_session_timeout_seconds=180,
    )

    async with chroma_mcp:
        triage = build_agents(chroma_mcp)
        tracer = globals().get("_otel_tracer")
        if tracer is None:
            result = await Runner.run(
                triage, question, hooks=hooks, max_turns=20
            )
        else:
            from openinference.semconv.trace import (
                OpenInferenceMimeTypeValues,
                OpenInferenceSpanKindValues,
                SpanAttributes,
            )

            with tracer.start_as_current_span("RAG request") as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.CHAIN.value,
                )
                span.set_attribute(SpanAttributes.INPUT_MIME_TYPE,
                                   OpenInferenceMimeTypeValues.TEXT.value)
                span.set_attribute(SpanAttributes.INPUT_VALUE, question)
                span.set_attribute("rag.vector_database", "ChromaDB")
                span.set_attribute("rag.chroma_db_dir", os.getenv(
                    "CHROMA_DB_DIR", str(PROJECT_DIR / "chroma_db")
                ))
                span.set_attribute("rag.chroma_collection", os.getenv(
                    "CHROMA_COLLECTION", "agentic_rag_docs"
                ))
                span.set_attribute("rag.mcp_server", "chroma_mcp_server.py")
                result = await Runner.run(
                    triage, question, hooks=hooks, max_turns=20
                )
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE,
                                   OpenInferenceMimeTypeValues.TEXT.value)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE,
                                   str(result.final_output))
        hooks.close_open_tool_spans()

    print("\n" + "=" * 78)
    print("FINAL ANSWER")
    print("=" * 78)
    print(result.final_output)

    # Fallback accounting if the SDK version has no on_llm_end hook
    if not any(s["kind"] == "llm" for s in tracker.steps):
        for raw in result.raw_responses:
            if raw.usage:
                tracker.add_llm("(run)", MODEL, raw.usage)

    print(tracker.report())

    provider = globals().get("_otel_provider")
    if provider is not None:
        provider.force_flush()
        print("\nTrace exported to AgentGuard.")


if __name__ == "__main__":
    asyncio.run(main())
