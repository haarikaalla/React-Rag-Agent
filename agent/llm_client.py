"""
LLM Client
Unified interface for Ollama (local), OpenAI, and Groq.
"""

from __future__ import annotations
from typing import List, Dict, Any
import os


def get_llm_client(
    provider: str,
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """Factory: return the right LangChain LLM object."""

    if not model_name or not model_name.strip():
        raise ValueError("Enter a model ID available from the selected provider.")

    p = provider.lower()

    if "ollama" in p:
        from langchain_community.llms import Ollama
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return Ollama(model=model_name, base_url=url)

    if "openai" in p:
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key is required.")
        return ChatOpenAI(model=model_name, api_key=key, temperature=0.1)

    if "groq" in p:
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Run: pip install langchain-groq")
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Groq API key is required.")
        return ChatGroq(model=model_name, api_key=key, temperature=0.1)

    raise ValueError(f"Unknown provider: {provider}. Choose Ollama, OpenAI, or Groq.")


def call_llm(llm, prompt: str) -> str:
    """
    Call any LangChain LLM (Chat or completion) and return text.
    Works for both ChatOpenAI (returns AIMessage) and Ollama (returns str).
    """
    response = llm.invoke(prompt)
    if hasattr(response, "content"):   # ChatModel
        return response.content.strip()
    return str(response).strip()
