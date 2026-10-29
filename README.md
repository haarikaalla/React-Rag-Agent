# ReAct RAG Agent

**Inspect document answers and the retrieval steps behind them**

[![CI](https://github.com/haarikaalla/React-Rag-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/haarikaalla/React-Rag-Agent/actions/workflows/ci.yml)

> Thought → Action[tool] → Observation → … → Final Answer

This project compares a single retrieval pass with a multi-step ReAct loop over
an uploaded PDF or text document. The agent chooses retrieval tools, records
observations, and shows the trace next to its answer. It abstains if retrieval
returns no evidence or the model tries to answer before retrieving. An LLM can
still make mistakes after seeing evidence, so check important answers against
the source document.

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

The retrieval gate sits between the agent loop and its final answer: a final
answer is displayed only after a successful retrieval observation. Simple RAG
skips the model call entirely when no chunks are returned. The Streamlit UI
renders document and model text as text, so uploaded content is not treated as
HTML.
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
| Retry path | No | Can use `clarify` to retrieve again |
| LLM providers | One | Ollama, OpenAI, Groq |
| Logging | None | JSONL session logs |
| Tests | None | 17 offline tests, including no-evidence behavior |
| Docker | No | Yes |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/haarikaalla/React-Rag-Agent.git
cd React-Rag-Agent
pip install -r requirements.txt
```

### 2. Run with Ollama (free, local)

```bash
# Install Ollama from its official site and start its local service
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
python -m pytest -q
```

---

## Usage

1. Upload a PDF or TXT document in the sidebar
2. Click **Build Knowledge Base** — chunks, embeds, indexes
3. Select LLM provider and configure API key if needed
4. Type your question in the chat input
5. Watch the agent's Thought → Action → Observation trace in real time
6. Toggle **Simple RAG** mode to compare single-shot vs multi-step

The UI displays the number of successful retrieval operations in ReAct mode.
In Simple RAG mode it displays the number of returned chunks. The displayed
token count is an estimate, not provider billing data.

## Verification and limits

The CI workflow runs the offline core tests on Python 3.10 and 3.13 and checks
that Streamlit starts on Python 3.13. Its
retrieval tests use a mock vector store, so they need no model download or API
key. For an end-to-end check, install the full requirements, upload a short
text file, build the knowledge base, ask a question it answers, then ask one it
does not answer. A live provider and local embedding download are needed for
that check; CI does not claim to exercise them.

The ReAct trace is useful for inspection but is not a proof that each answer
sentence is supported. This app has no document permission model or persisted vector index,
or evaluation benchmark. Use approved providers and documents when sending
private content to a cloud model. Ollama keeps inference local when configured
against a local service. Queries and answers are written to local `logs/`
JSONL files; treat them as sensitive when working with private documents.

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

- **LangChain** — LLM calls and text splitting
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


