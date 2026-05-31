"""
ReAct Agent
Implements the Reasoning + Acting loop:
  Thought → Action[tool, input] → Observation → … → Final Answer

Falls back to simple RAG if use_react=False.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from agent.llm_client import get_llm_client, call_llm
from rag.pipeline import RAGPipeline
from tools.tool_registry import ToolRegistry


# ── Prompt templates ────────────────────────────────────────────────────

REACT_SYSTEM = """You are a precise question-answering agent with access to a knowledge base.
You reason step-by-step using the ReAct pattern.

TOOLS AVAILABLE:
{tools}

FORMAT — follow this EXACTLY for each step:
Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input to the tool>

After receiving an Observation, continue with another Thought/Action or end with:
Thought: I have enough information to answer.
Final Answer: <your complete, grounded answer>

RULES:
- Only use facts from Observations. Never hallucinate.
- If Observations do not contain the answer, say "The document does not contain information about this."
- Use the retrieve tool first; use clarify if the first retrieval was insufficient.
- Stop after {max_steps} Action steps even if uncertain; give best answer from evidence so far.
- Keep Final Answer concise and directly responsive to the question.

QUESTION: {question}

Begin.
"""

SIMPLE_RAG_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the context below. Do not hallucinate.
If the context does not contain the answer, say so explicitly.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""


# ── Parser helpers ────────────────────────────────────────────────────────

def _parse_action(text: str):
    """Extract (tool_name, action_input) from LLM output."""
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if action_match and input_match:
        return action_match.group(1).strip(), input_match.group(1).strip()
    return None, None


def _parse_final_answer(text: str):
    m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return everything after last "Thought:"
    parts = text.rsplit("Thought:", 1)
    return parts[-1].strip() if parts else text.strip()


def _extract_thought(text: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?:\nAction:|Final Answer:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── ReAct Agent ───────────────────────────────────────────────────────────

class ReActAgent:
    """
    Multi-step RAG agent with ReAct reasoning loop.

    Args:
        pipeline:   Initialised RAGPipeline
        provider:   "Ollama (Local)" | "OpenAI" | "Groq"
        model_name: Model identifier for the chosen provider
        api_key:    API key (None for Ollama)
        base_url:   Ollama base URL (None for cloud providers)
        max_steps:  Maximum Thought/Action/Observation iterations
        use_react:  True for full ReAct loop, False for single-shot RAG
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        provider: str,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_steps: int = 4,
        use_react: bool = True,
    ):
        self.pipeline   = pipeline
        self.max_steps  = max_steps
        self.use_react  = use_react
        self.tools      = ToolRegistry(pipeline)
        self.llm        = get_llm_client(provider, model_name, api_key, base_url)

    def run(self, question: str) -> Dict[str, Any]:
        if not self.use_react:
            return self._simple_rag(question)
        return self._react_loop(question)

    # ── ReAct loop ────────────────────────────────────────────────────────

    def _react_loop(self, question: str) -> Dict[str, Any]:
        trace: List[Dict[str, str]] = []
        total_tokens = 0

        prompt = REACT_SYSTEM.format(
            tools=self.tools.tool_descriptions(),
            max_steps=self.max_steps,
            question=question,
        )
        scratchpad = ""

        for step in range(self.max_steps):
            full_prompt = prompt + scratchpad
            response = call_llm(self.llm, full_prompt)

            # Track token estimate (rough: 1 token ≈ 4 chars)
            total_tokens += len(response) // 4

            # Check for Final Answer
            if "Final Answer:" in response:
                thought = _extract_thought(response)
                if thought:
                    trace.append({"type": "thought", "content": thought})
                answer = _parse_final_answer(response)
                return {
                    "answer":       answer,
                    "trace":        trace,
                    "num_sources":  self._count_retrieved(trace),
                    "total_tokens": total_tokens,
                    "steps":        step + 1,
                }

            # Parse Action
            tool_name, tool_input = _parse_action(response)
            thought = _extract_thought(response)

            if thought:
                trace.append({"type": "thought", "content": thought})

            if not tool_name:
                # No parseable action — treat remaining text as final answer
                answer = response.split("Thought:")[-1].strip()
                return {
                    "answer":       answer or "Could not determine an answer.",
                    "trace":        trace,
                    "num_sources":  self._count_retrieved(trace),
                    "total_tokens": total_tokens,
                    "steps":        step + 1,
                }

            trace.append({"type": "action", "tool": tool_name, "content": tool_input})

            # Call tool
            observation = self.tools.call(tool_name, tool_input)
            trace.append({"type": "observation", "content": observation[:800]})  # truncate for display

            # Extend scratchpad
            scratchpad += (
                f"\nThought: {thought}"
                f"\nAction: {tool_name}"
                f"\nAction Input: {tool_input}"
                f"\nObservation: {observation}\n"
            )

        # Max steps reached — ask LLM to conclude from scratchpad
        conclude_prompt = (
            prompt + scratchpad +
            "\nThought: I have reached the step limit. I will answer from the evidence gathered.\nFinal Answer:"
        )
        final = call_llm(self.llm, conclude_prompt)
        answer = final.split("Final Answer:")[-1].strip() if "Final Answer:" in final else final.strip()

        return {
            "answer":       answer or "Reached step limit without a confident answer.",
            "trace":        trace,
            "num_sources":  self._count_retrieved(trace),
            "total_tokens": total_tokens,
            "steps":        self.max_steps,
        }

    # ── Simple RAG (no loop) ──────────────────────────────────────────────

    def _simple_rag(self, question: str) -> Dict[str, Any]:
        context = self.pipeline.get_context(question)
        prompt  = SIMPLE_RAG_PROMPT.format(context=context, question=question)
        answer  = call_llm(self.llm, prompt)
        docs    = self.pipeline.retrieve(question)
        return {
            "answer":       answer,
            "trace":        [],
            "num_sources":  len(docs),
            "total_tokens": len(prompt) // 4 + len(answer) // 4,
            "steps":        1,
        }

    # ── Helper ───────────────────────────────────────────────────────────

    def _count_retrieved(self, trace: List[Dict]) -> int:
        return sum(1 for s in trace if s["type"] == "observation")
