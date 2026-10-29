"""
Unit tests — run with: pytest tests/ -v
Tests run without an LLM (mocked) and without a real document.
"""

import pytest
from unittest.mock import MagicMock, patch
import io


# ── RAG Pipeline ─────────────────────────────────────────────────────────

class TestRAGPipeline:
    def _make_pipeline(self):
        from rag.pipeline import RAGPipeline
        return RAGPipeline(chunk_size=200, chunk_overlap=20, top_k=2)

    def test_init(self):
        p = self._make_pipeline()
        assert p.chunk_size == 200
        assert p.num_chunks == 0
        assert p._vectorstore is None

    def test_ingest_txt(self):
        p = self._make_pipeline()
        content = "Machine learning learns from data."
        fake_file = MagicMock()
        fake_file.name = "test.txt"
        fake_file.read.return_value = content.encode("utf-8")
        with patch.object(p, "_split", return_value=["chunk"]) as split, patch.object(p, "_build_index") as build:
            p.ingest(fake_file)
        split.assert_called_once_with(content)
        build.assert_called_once_with(["chunk"])
        assert p.num_chunks == 1

    def test_retrieve_after_ingest(self):
        p = self._make_pipeline()
        doc = MagicMock(page_content="Python is a programming language.", metadata={"source": "test.txt"})
        p._vectorstore = MagicMock()
        p._vectorstore.similarity_search_with_score.return_value = [(doc, 0.2)]
        results = p.retrieve("What is Python?")
        assert results == [{"content": "Python is a programming language.", "score": 0.2, "source": "test.txt"}]

    def test_retrieve_before_ingest_raises(self):
        p = self._make_pipeline()
        with pytest.raises(RuntimeError, match="not initialised"):
            p.retrieve("test query")

    def test_get_context_returns_string(self):
        p = self._make_pipeline()
        p.retrieve = MagicMock(return_value=[{"content": "LangChain is a framework.", "score": 0.1}])
        ctx = p.get_context("What is LangChain?")
        assert "LangChain is a framework." in ctx


# ── Tool Registry ────────────────────────────────────────────────────────

class TestToolRegistry:
    def _make_registry(self):
        from tools.tool_registry import ToolRegistry
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            {"content": "FAISS is a vector similarity library.", "score": 0.25, "source": "doc"}
        ]
        return ToolRegistry(mock_pipeline)

    def test_default_tools_registered(self):
        reg = self._make_registry()
        assert "retrieve" in reg.names
        assert "summarise" in reg.names
        assert "clarify" in reg.names

    def test_retrieve_tool_returns_string(self):
        reg = self._make_registry()
        result = reg.call("retrieve", "What is FAISS?")
        assert isinstance(result, str)
        assert "FAISS" in result

    def test_unknown_tool_returns_error(self):
        reg = self._make_registry()
        result = reg.call("nonexistent_tool", "query")
        assert "ToolError" in result

    def test_tool_descriptions_nonempty(self):
        reg = self._make_registry()
        desc = reg.tool_descriptions()
        assert "retrieve" in desc
        assert len(desc) > 20


# ── ReAct Agent (mocked LLM) ─────────────────────────────────────────────

class TestReActAgent:
    def _make_agent(self, llm_response: str, use_react: bool = True):
        from agent.react_agent import ReActAgent
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            {"content": "The sky is blue due to Rayleigh scattering.", "score": 0.2, "source": "doc"}
        ]
        mock_pipeline.get_context.return_value = "The sky is blue due to Rayleigh scattering."

        agent = ReActAgent.__new__(ReActAgent)
        agent.pipeline  = mock_pipeline
        agent.max_steps = 3
        agent.use_react = use_react

        from tools.tool_registry import ToolRegistry
        agent.tools = ToolRegistry(mock_pipeline)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = llm_response
        agent.llm = mock_llm
        return agent

    def test_simple_rag_returns_answer(self):
        agent = self._make_agent("The sky is blue.", use_react=False)
        result = agent.run("Why is the sky blue?")
        assert "answer" in result
        assert result["answer"] == "The sky is blue."

    def test_simple_rag_abstains_without_retrieval(self):
        from agent.react_agent import ABSTAIN
        agent = self._make_agent("Invented answer", use_react=False)
        agent.pipeline.retrieve.return_value = []
        result = agent.run("What is not in the document?")
        assert result["answer"] == ABSTAIN
        assert result["num_sources"] == 0
        agent.llm.invoke.assert_not_called()

    def test_react_abstains_on_unsupported_final_answer(self):
        from agent.react_agent import ABSTAIN
        agent = self._make_agent("Thought: done.\nFinal Answer: Invented answer.")
        result = agent.run("unknown?")
        assert result["answer"] == ABSTAIN
        assert result["num_sources"] == 0

    def test_react_abstains_after_empty_retrieval(self):
        from agent.react_agent import ABSTAIN
        agent = self._make_agent("Action: retrieve\nAction Input: unknown")
        agent.pipeline.retrieve.return_value = []
        result = agent.run("unknown?")
        assert result["answer"] == ABSTAIN
        assert result["num_sources"] == 0

    def test_react_final_answer_parsed(self):
        llm_resp = (
            "Thought: I should retrieve information.\n"
            "Action: retrieve\n"
            "Action Input: sky blue color\n"
        )
        # Second call returns Final Answer
        from agent.react_agent import ReActAgent
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = [
            {"content": "Blue light scatters more due to Rayleigh scattering.", "score": 0.1, "source": "doc"}
        ]
        mock_pipeline.get_context.return_value = "Blue light scatters more."

        agent = ReActAgent.__new__(ReActAgent)
        agent.pipeline  = mock_pipeline
        agent.max_steps = 4
        agent.use_react = True
        from tools.tool_registry import ToolRegistry
        agent.tools = ToolRegistry(mock_pipeline)

        call_count = [0]
        def side_effect(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                r = MagicMock()
                r.content = llm_resp
                return r
            r = MagicMock()
            r.content = "Thought: I have enough info.\nFinal Answer: The sky is blue due to Rayleigh scattering."
            return r

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = side_effect
        agent.llm = mock_llm

        result = agent.run("Why is sky blue?")
        assert "Rayleigh" in result["answer"]
        assert len(result["trace"]) > 0

    def test_result_has_required_keys(self):
        agent = self._make_agent(
            "Thought: done.\nFinal Answer: Test answer.",
            use_react=True
        )
        result = agent.run("test?")
        for key in ["answer", "trace", "num_sources", "total_tokens"]:
            assert key in result


# ── Agent Logger ──────────────────────────────────────────────────────────

class TestAgentLogger:
    def test_log_and_summary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from utils.logger import AgentLogger
        logger = AgentLogger()
        logger.log("q1", {"answer": "a1", "steps": 2, "num_sources": 3, "total_tokens": 50}, 1.2)
        logger.log("q2", {"answer": "a2", "steps": 3, "num_sources": 2, "total_tokens": 80}, 0.8)

        summary = logger.summary()
        assert summary["total_queries"] == 2
        assert summary["avg_latency_s"] == 1.0
        assert summary["total_tokens"] == 130


def test_missing_model_id_has_clear_error():
    from agent.llm_client import get_llm_client

    with pytest.raises(ValueError, match="model ID"):
        get_llm_client("Groq", "")
