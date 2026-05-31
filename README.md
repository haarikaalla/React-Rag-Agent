# ReAct RAG Agent

**Multi-step Retrieval-Augmented Generation with ReAct reasoning loop**

> Thought → Action[tool] → Observation → … → Final Answer

A production-grade agentic RAG system that goes beyond simple retrieve-then-answer. The agent *reasons* about what to retrieve, *selects* tools dynamically, *reflects* on observations, and iterates until it has enough evidence to give a grounded answer — all visible in real time.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              ReAct Agent Loop               │
│                                             │
│  Thought → Action → Observation → Thought  │
│       ↑_________________________________↑  │
│                                             │
│  Tools:  retrieve │ summarise │ clarify     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              RAG Pipeline                   │
│  PDF/TXT → Chunker → Embeddings → FAISS    │
│  (sentence-transformers/all-MiniLM-L6-v2)  │
└─────────────────────────────────────────────┘
               │
               ▼
         LLM (Ollama / OpenAI / Groq)
```

**Key components:**
- `agent/react_agent.py` — ReAct loop with Thought/Action/Observation parsing
- `rag/pipeline.py` — Document ingestion, chunking, embedding, FAISS retrieval
- `tools/tool_registry.py` — Pluggable tool registry (retrieve, summarise, clarify)
- `utils/logger.py` — JSONL session logger for eval and benchmarking
- `app.py` — Streamlit UI with live trace visualisation

---

## What makes this different from a basic RAG chatbot

| Feature | Basic RAG | This Project |
|---|---|---|
| Reasoning | Single retrieve → answer | Multi-step Thought/Action loop |
| Tool use | None | retrieve, summarise, clarify |
| Trace visibility | Hidden | Full step-by-step UI |
| Self-correction | No | Re-retrieves if first pass insufficient |
| LLM providers | One | Ollama, OpenAI, Groq |
| Logging | None | JSONL session logs |
| Tests | None | pytest suite |
| Docker | No | Yes |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/haarikaalla/react-rag-agent.git
cd react-rag-agent
pip install -r requirements.txt
```

### 2. Run with Ollama (free, local)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
streamlit run app.py
```

### 3. Run with OpenAI

```bash
# Set your key in the sidebar or:
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

### 4. Run with Docker

```bash
docker build -t react-rag-agent .
docker run -p 8501:8501 react-rag-agent
# Open http://localhost:8501
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Usage

1. Upload a PDF or TXT document in the sidebar
2. Click **Build Knowledge Base** — chunks, embeds, indexes
3. Select LLM provider and configure API key if needed
4. Type your question in the chat input
5. Watch the agent's Thought → Action → Observation trace in real time
6. Toggle **Simple RAG** mode to compare single-shot vs multi-step

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| Chunk Size | 400 | Characters per chunk |
| Chunk Overlap | 60 | Overlap between chunks |
| Top-K | 3 | Chunks retrieved per tool call |
| Max Steps | 4 | Max Thought/Action iterations |

---

## Tech Stack

- **LangChain** — LLM orchestration, text splitting
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Local embeddings (all-MiniLM-L6-v2)
- **Streamlit** — UI and live trace display
- **PyMuPDF** — PDF text extraction
- **Ollama / OpenAI / Groq** — LLM inference

---

## Project Structure

```
react-rag-agent/
├── app.py                  # Streamlit UI
├── agent/
│   ├── react_agent.py      # ReAct loop + prompt templates
│   └── llm_client.py       # Unified LLM interface
├── rag/
│   └── pipeline.py         # Ingestion → chunking → embedding → FAISS
├── tools/
│   └── tool_registry.py    # Pluggable agent tools
├── utils/
│   └── logger.py           # JSONL session logger
├── tests/
│   └── test_all.py         # pytest suite
├── Dockerfile
└── requirements.txt
```

---


