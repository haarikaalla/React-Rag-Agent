"""
Agent Tools
Each tool takes a query string and returns an observation string.
The agent calls these based on its Thought → Action decisions.
"""

from __future__ import annotations
from typing import Dict, Callable, Any
from rag.pipeline import RAGPipeline


class ToolRegistry:
    """Holds all tools available to the agent."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(
            name="retrieve",
            description=(
                "Search the knowledge base for relevant passages. "
                "Use when you need factual information from the document. "
                "Input: a search query string."
            ),
            fn=self._tool_retrieve,
        )
        self.register(
            name="summarise",
            description=(
                "Retrieve and summarise the most relevant sections for a topic. "
                "Use when the question is broad or requires an overview. "
                "Input: a topic or concept string."
            ),
            fn=self._tool_summarise,
        )
        self.register(
            name="clarify",
            description=(
                "Re-retrieve with a refined or more specific query if the first "
                "retrieval was insufficient. Input: a more specific search string."
            ),
            fn=self._tool_clarify,
        )

    def register(self, name: str, description: str, fn: Callable) -> None:
        self._tools[name] = {"description": description, "fn": fn}

    def call(self, name: str, input_str: str) -> str:
        if name not in self._tools:
            available = ", ".join(self._tools.keys())
            return f"[ToolError] Unknown tool '{name}'. Available: {available}"
        try:
            return self._tools[name]["fn"](input_str)
        except Exception as e:
            return f"[ToolError] {name} failed: {e}"

    def tool_descriptions(self) -> str:
        lines = []
        for name, meta in self._tools.items():
            lines.append(f"- {name}: {meta['description']}")
        return "\n".join(lines)

    @property
    def names(self):
        return list(self._tools.keys())

    # ── Tool implementations ────────────────────────────────────────────

    def _tool_retrieve(self, query: str) -> str:
        docs = self.pipeline.retrieve(query)
        if not docs:
            return "No relevant content found for that query."
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[Chunk {i} | score={d['score']}]\n{d['content']}")
        return "\n\n".join(parts)

    def _tool_summarise(self, topic: str) -> str:
        docs = self.pipeline.retrieve(topic, top_k=4)
        if not docs:
            return "No content found for that topic."
        combined = " ".join(d["content"] for d in docs)
        # Truncate for prompt safety
        return combined[:2000]

    def _tool_clarify(self, refined_query: str) -> str:
        docs = self.pipeline.retrieve(refined_query, top_k=3)
        if not docs:
            return "Still no relevant content found."
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[Chunk {i}]\n{d['content']}")
        return "\n\n".join(parts)
