"""
RAG Pipeline
Handles: document ingestion → chunking → embedding → FAISS indexing → retrieval
"""

from __future__ import annotations
from typing import List, Dict, Any


def _extract_text(uploaded_file) -> str:
    """Extract raw text from PDF or TXT upload."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw, filetype="pdf")
            return "\n".join(page.get_text() for page in doc).strip()
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Run: pip install pymupdf"
            )

    # Plain text / markdown
    return raw.decode("utf-8", errors="ignore").strip()


class RAGPipeline:
    """
    Full RAG pipeline with configurable chunking and retrieval.

    Usage:
        pipeline = RAGPipeline(chunk_size=400, chunk_overlap=60, top_k=3)
        pipeline.ingest(uploaded_file)
        docs = pipeline.retrieve("What is X?")
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        top_k: int = 3,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k
        self.num_chunks    = 0
        self._vectorstore  = None
        self._embeddings   = None

    # ── Public API ───────────────────────────────────────────────────────

    def ingest(self, uploaded_file) -> None:
        """Ingest a document: extract → split → embed → index."""
        raw_text = _extract_text(uploaded_file)
        if not raw_text:
            raise ValueError("Document appears to be empty or unreadable.")

        chunks = self._split(raw_text)
        self.num_chunks = len(chunks)
        self._build_index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return top-k chunks with similarity scores."""
        if self._vectorstore is None:
            raise RuntimeError("Pipeline not initialised. Call ingest() first.")

        k = top_k or self.top_k
        results = self._vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "score":   round(float(score), 4),
                "source":  doc.metadata.get("source", "document"),
            }
            for doc, score in results
        ]

    def get_context(self, query: str, top_k: int | None = None) -> str:
        """Return retrieved chunks as a single formatted context string."""
        docs = self.retrieve(query, top_k=top_k)
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[Source {i} | score={d['score']}]\n{d['content']}")
        return "\n\n---\n\n".join(parts)

    # ── Private helpers ──────────────────────────────────────────────────

    def _split(self, text: str) -> List[Document]:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata={"source": "document", "chunk_id": i})
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]

    def _build_index(self, chunks: List[Document]) -> None:
        from langchain_community.vectorstores import FAISS

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
