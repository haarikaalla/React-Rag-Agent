"""
ReAct RAG Agent — Streamlit UI
Multi-step reasoning: Plan → Retrieve → Reflect → Answer
"""

import streamlit as st
from agent.react_agent import ReActAgent
from rag.pipeline import RAGPipeline
from utils.logger import AgentLogger
import time

st.set_page_config(
    page_title="ReAct RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.thought-box  { background:#f0f4ff; border-left:4px solid #3b82f6; padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.88rem; }
.action-box   { background:#fff7ed; border-left:4px solid #f97316; padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.88rem; }
.obs-box      { background:#f0fdf4; border-left:4px solid #22c55e; padding:10px 14px; border-radius:6px; margin:6px 0; font-size:0.88rem; }
.answer-box   { background:#faf5ff; border-left:4px solid #a855f7; padding:12px 16px; border-radius:6px; margin:8px 0; font-size:0.95rem; font-weight:500; }
.metric-card  { background:#f8fafc; border:1px solid #e2e8f0; padding:8px 12px; border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "logger" not in st.session_state:
    st.session_state.logger = AgentLogger()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Agent Config")
    st.divider()

    uploaded_file = st.file_uploader("📄 Upload Document (PDF / TXT)", type=["pdf", "txt"])

    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["Ollama (Local)", "OpenAI", "Groq"])

    if provider == "Ollama (Local)":
        model_name = st.selectbox("Model", ["llama3", "mistral", "phi3", "gemma2"])
        api_key = None
        base_url = st.text_input("Ollama URL", value="http://localhost:11434")
    elif provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        api_key = st.text_input("OpenAI API Key", type="password")
        base_url = None
    else:  # Groq
        model_name = st.selectbox("Model", ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"])
        api_key = st.text_input("Groq API Key", type="password")
        base_url = None

    st.subheader("RAG Settings")
    chunk_size   = st.slider("Chunk Size",   128, 1024, 400, step=64)
    chunk_overlap= st.slider("Chunk Overlap", 0,  256,  60,  step=16)
    top_k        = st.slider("Top-K Docs",    1,  8,    3)
    max_steps    = st.slider("Max Agent Steps", 2, 8, 4)

    st.subheader("Agent Mode")
    agent_mode = st.selectbox("Reasoning", ["ReAct (Reason+Act)", "Simple RAG"])
    show_trace = st.checkbox("Show Agent Trace", value=True)

    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.logger = AgentLogger()
        st.rerun()

    if uploaded_file and st.button("⚡ Build Knowledge Base", type="primary"):
        with st.spinner("Chunking, embedding, indexing…"):
            try:
                pipeline = RAGPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
                pipeline.ingest(uploaded_file)
                st.session_state.pipeline = pipeline
                st.success(f"✅ Indexed {pipeline.num_chunks} chunks")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# ── Main panel ───────────────────────────────────────────────────────────
st.title("🤖 ReAct RAG Agent")
st.caption("Multi-step reasoning: **Thought → Action → Observation → Answer**")

# Stats row
if st.session_state.pipeline:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks Indexed", st.session_state.pipeline.num_chunks)
    c2.metric("Queries Run",    len(st.session_state.chat_history))
    c3.metric("Mode",           "ReAct" if agent_mode.startswith("ReAct") else "Simple")
    c4.metric("Model",          model_name)
    st.divider()

# Chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        if show_trace and entry.get("trace"):
            with st.expander("🧠 Agent Trace", expanded=False):
                for step in entry["trace"]:
                    if step["type"] == "thought":
                        st.markdown(f'<div class="thought-box">💭 <b>Thought:</b> {step["content"]}</div>', unsafe_allow_html=True)
                    elif step["type"] == "action":
                        st.markdown(f'<div class="action-box">⚡ <b>Action [{step.get("tool","")}]:</b> {step["content"]}</div>', unsafe_allow_html=True)
                    elif step["type"] == "observation":
                        st.markdown(f'<div class="obs-box">👁️ <b>Observation:</b> {step["content"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">✅ {entry["answer"]}</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].caption(f"⏱ {entry.get('latency', 0):.2f}s")
        cols[1].caption(f"📄 {entry.get('sources', 0)} sources")
        cols[2].caption(f"🔢 {entry.get('tokens', '—')} tokens")

# Query input
if not st.session_state.pipeline:
    st.info("👆 Upload a document and click **Build Knowledge Base** to start.")
else:
    query = st.chat_input("Ask anything about your document…")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            trace_placeholder = st.empty()
            answer_placeholder = st.empty()

            with st.spinner("Agent thinking…"):
                try:
                    agent = ReActAgent(
                        pipeline=st.session_state.pipeline,
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url if provider == "Ollama (Local)" else None,
                        max_steps=max_steps,
                        use_react=(agent_mode.startswith("ReAct")),
                    )
                    t0 = time.time()
                    result = agent.run(query)
                    latency = time.time() - t0

                    if show_trace and result.get("trace"):
                        with trace_placeholder.expander("🧠 Agent Trace", expanded=True):
                            for step in result["trace"]:
                                if step["type"] == "thought":
                                    st.markdown(f'<div class="thought-box">💭 <b>Thought:</b> {step["content"]}</div>', unsafe_allow_html=True)
                                elif step["type"] == "action":
                                    st.markdown(f'<div class="action-box">⚡ <b>Action [{step.get("tool","")}]:</b> {step["content"]}</div>', unsafe_allow_html=True)
                                elif step["type"] == "observation":
                                    st.markdown(f'<div class="obs-box">👁️ <b>Observation:</b> {step["content"]}</div>', unsafe_allow_html=True)

                    answer_placeholder.markdown(
                        f'<div class="answer-box">✅ {result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.chat_history.append({
                        "question": query,
                        "answer":   result["answer"],
                        "trace":    result.get("trace", []),
                        "latency":  latency,
                        "sources":  result.get("num_sources", 0),
                        "tokens":   result.get("total_tokens", "—"),
                    })
                    st.session_state.logger.log(query, result, latency)

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    raise
