"""The RAG system under test.

The harness only ever calls RagPipeline.answer(). Everything inside this
file is swappable — that separation is the whole point of the project.

Uses the google-genai SDK rather than the deprecated google-generativeai.
The reason is not just the deprecation warning: the old package configured
the API key globally for the whole process, so two concurrent runs with
different user-supplied keys would overwrite each other's credentials. A
Client object is per-instance, which is what bring-your-own-key needs.
"""

import os
import time
from typing import Optional

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from evalharness.pipeline.loader import Chunk, load_chunks
from evalharness.schemas import PipelineOutput, RetrievedChunk

import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"

ANSWER_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say so plainly — do not guess.
Be concise: two or three sentences.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class RagPipeline:
    def __init__(
        self,
        docs_dir: str = "data/docs",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 5,
        gen_model: str = "gemini-3.1-flash-lite",
        persist_dir: str = ".chroma",
        collection_name: str = "baseline",
        api_key: Optional[str] = None,
    ):
        # Per-instance client. Two pipelines with different keys never clash.
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

        self.top_k = top_k
        self.gen_model = gen_model

        self.chunks: list[Chunk] = load_chunks(docs_dir, chunk_size, chunk_overlap)

        chroma = chromadb.PersistentClient(path=persist_dir)
        # Rebuild each time so config changes (chunk_size!) always take effect.
        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = chroma.create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._index()

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed a batch of texts.

        task_type matters: documents and queries get embedded differently by
        this model, and using the wrong one measurably hurts retrieval.
        """
        result = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]

    def _index(self) -> None:
        """Embed all chunks and load them into Chroma."""
        texts = [c.text for c in self.chunks]
        embeddings = self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

        self.collection.add(
            ids=[c.doc_id for c in self.chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c.source, "index": c.index} for c in self.chunks],
        )
        print(f"Indexed {len(self.chunks)} chunks into '{self.collection.name}'")

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        q_emb = self._embed([question], task_type="RETRIEVAL_QUERY")[0]

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(self.top_k, len(self.chunks)),
        )

        out: list[RetrievedChunk] = []
        for rank, (doc_id, text, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["distances"][0]), start=1
        ):
            out.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    text=text,
                    score=1.0 - dist,   # cosine distance -> similarity
                    rank=rank,          # 1-indexed, matters for MRR
                )
            )
        return out

    def answer(self, question: str, test_case_id: str = "adhoc") -> PipelineOutput:
        """The only method the harness calls."""
        start = time.perf_counter()

        retrieved = self.retrieve(question)
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in retrieved)

        resp = self.client.models.generate_content(
            model=self.gen_model,
            contents=ANSWER_PROMPT.format(
                context=context or "(nothing retrieved)", question=question
            ),
            config=types.GenerateContentConfig(temperature=0.0),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = getattr(resp, "usage_metadata", None)
        return PipelineOutput(
            test_case_id=test_case_id,
            answer=(resp.text or "").strip(),
            retrieved=retrieved,
            latency_ms=elapsed_ms,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


if __name__ == "__main__":
    pipe = RagPipeline()
    q = "How many days of paid leave do employees get?"

    out = pipe.answer(q)
    print(f"\nQ: {q}")
    print(f"A: {out.answer}")
    print(f"\nlatency: {out.latency_ms:.0f}ms | tokens: {out.prompt_tokens}+{out.completion_tokens}")
    print("retrieved:")
    for c in out.retrieved:
        print(f"  {c.rank}. {c.doc_id}  (score {c.score:.3f})")