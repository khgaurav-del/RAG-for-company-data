
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

import spacy
nlp = spacy.load('en_core_web_md')

try:
    from pymongo import ASCENDING, MongoClient, ReplaceOne
except Exception:
    ASCENDING = None
    MongoClient = None
    ReplaceOne = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None


GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def recursive_chunking(text: str, target_size: int = 800) -> List[str]:
    sections = text.split("\n## ")
    chunks: List[str] = []
    for section in sections:
        if len(section) > 1000:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current + para) < target_size:
                    current += para + "\n\n"
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = para + "\n\n"
            if current.strip():
                chunks.append(current.strip())
        elif section.strip():
            chunks.append(section.strip())
    return chunks


def Semantic_chunking(text, max_chars=800):
    try:
       
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        # Fallback if spaCy model is unavailable
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def read_corpus_documents(corpus_path: str) -> List[Dict[str, str]]:
    path = Path(corpus_path)
    docs: List[Dict[str, str]] = []

    def load_text_file(fp: Path) -> None:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": str(fp), "text": text})

    def load_csv_file(fp: Path) -> None:
        df = pd.read_csv(fp)
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not text_cols:
            text_cols = list(df.columns)

        for idx, row in df.iterrows():
            row_parts = []
            for col in text_cols:
                value = row.get(col, None)
                if pd.notna(value):
                    s = str(value).strip()
                    if s:
                        row_parts.append(f"{col}: {s}")
            if row_parts:
                docs.append({"source": f"{fp}#row={idx}", "text": "\n".join(row_parts)})

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in [".txt", ".md"]:
            load_text_file(path)
        elif suffix == ".csv":
            load_csv_file(path)
        return docs

    if path.is_dir():
        for pattern in ["*.txt", "*.md"]:
            for fp in path.rglob(pattern):
                load_text_file(fp)
        for fp in path.rglob("*.csv"):
            load_csv_file(fp)

    return docs


def build_chunks(docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    cid = 0
    for doc in docs:
        local_chunks = Semantic_chunking(doc["text"])
        for chunk_text in local_chunks:
            chunks.append({"id": f"ch_{cid}", "text": chunk_text, "source": doc["source"]})
            cid += 1
    return chunks


def get_mongo_collection():
    mongo_uri = os.getenv("MONGODB_URI", "").strip()
    if not mongo_uri or MongoClient is None:
        return None

    db_name = os.getenv("MONGODB_DB", "rag_db")
    coll_name = os.getenv("MONGODB_COLLECTION", "rag_chunks")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        collection = client[db_name][coll_name]
        if ASCENDING is not None:
            collection.create_index([("chunk_id", ASCENDING)], unique=True)
            collection.create_index([("source", ASCENDING)])
        return collection
    except Exception:
        return None


def upsert_chunks_to_mongodb(collection, chunks: List[Dict[str, str]]) -> None:
    if collection is None or ReplaceOne is None or not chunks:
        return

    ops = []
    for chunk in chunks:
        payload = {
            "chunk_id": chunk["id"],
            "text": chunk["text"],
            "source": chunk.get("source", "unknown"),
        }
        ops.append(ReplaceOne({"chunk_id": chunk["id"]}, payload, upsert=True))

    if ops:
        collection.bulk_write(ops, ordered=False)


def load_chunks_from_mongodb(collection, limit: int = 20000) -> List[Dict[str, str]]:
    if collection is None:
        return []

    records = list(collection.find({}, {"_id": 0, "chunk_id": 1, "text": 1, "source": 1}).limit(limit))
    chunks = []
    for doc in records:
        chunk_id = str(doc.get("chunk_id", "")).strip()
        text = str(doc.get("text", "")).strip()
        source = str(doc.get("source", "unknown")).strip() or "unknown"
        if chunk_id and text:
            chunks.append({"id": chunk_id, "text": text, "source": source})
    return chunks


class HybridRetriever:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25_model = None
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.local_chunk_matrix = None
        self.pc_client = None
        self.pc_index = None
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-assignment3-index")

    def set_corpus(self, chunks: List[Dict[str, Any]]) -> None:
        self.bm25_chunks = chunks
        if BM25Okapi is not None:
            tokenized = [c["text"].lower().split() for c in chunks]
            self.bm25_model = BM25Okapi(tokenized)
        vectors = self.embedding_model.encode([c["text"] for c in chunks], show_progress_bar=False)
        self.local_chunk_matrix = np.array(vectors)

    def try_init_pinecone(self) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        region = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        if not (Pinecone and api_key):
            return

        try:
            self.pc_client = Pinecone(api_key=api_key)
            existing = [idx.name for idx in self.pc_client.list_indexes()]
            if self.index_name not in existing:
                self.pc_client.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=region),
                )
            self.pc_index = self.pc_client.Index(self.index_name)
        except Exception:
            self.pc_client = None
            self.pc_index = None

    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        if self.pc_index is None:
            return
        vectors = []
        for chunk in chunks:
            vec = self.embedding_model.encode(chunk["text"]).tolist()
            vectors.append(
                {
                    "id": chunk["id"],
                    "values": vec,
                    "metadata": {"text": chunk["text"], "source": chunk["source"]},
                }
            )
        for i in range(0, len(vectors), batch_size):
            self.pc_index.upsert(vectors=vectors[i : i + batch_size])

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25_model is None:
            return []
        scores = self.bm25_model.get_scores(query.lower().split())
        indices = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in indices:
            doc = dict(self.bm25_chunks[i])
            doc["score"] = float(scores[i])
            doc["search_type"] = "keyword"
            out.append(doc)
        return out

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.pc_index is not None:
            try:
                qv = self.embedding_model.encode(query).tolist()
                response = self.pc_index.query(vector=qv, top_k=top_k, include_metadata=True)
                out = []
                for m in response.matches:
                    meta = m.metadata or {}
                    out.append(
                        {
                            "id": m.id,
                            "text": meta.get("text", ""),
                            "source": meta.get("source", "unknown"),
                            "score": float(m.score),
                            "search_type": "semantic",
                        }
                    )
                return out
            except Exception:
                pass

        if self.local_chunk_matrix is None or len(self.bm25_chunks) == 0:
            return []

        qv = self.embedding_model.encode(query)
        sims = cosine_similarity([qv], self.local_chunk_matrix)[0]
        indices = np.argsort(sims)[::-1][:top_k]
        out = []
        for i in indices:
            doc = dict(self.bm25_chunks[i])
            doc["score"] = float(sims[i])
            doc["search_type"] = "semantic"
            out.append(doc)
        return out

    @staticmethod
    def _rrf_fusion(keyword_results: List[Dict[str, Any]], semantic_results: List[Dict[str, Any]], k: int = 60):
        scores: Dict[str, float] = {}
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(keyword_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            merged[doc_id] = doc

        for rank, doc in enumerate(semantic_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id in merged:
                merged[doc_id]["search_type"] = "hybrid"
            else:
                merged[doc_id] = doc

        fused = []
        for doc_id, score in scores.items():
            d = dict(merged[doc_id])
            d["rrf_score"] = score
            fused.append(d)

        fused.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return fused

    def _rerank(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not results:
            return []
        qv = self.embedding_model.encode(query)
        reranked = []
        for doc in results:
            dv = self.embedding_model.encode(doc.get("text", ""))
            sim = float(cosine_similarity([qv], [dv])[0][0])
            d = dict(doc)
            d["rerank_score"] = sim
            reranked.append(d)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        keyword = self._bm25_search(query, top_k=8)
        semantic = self._semantic_search(query, top_k=8)
        fused = self._rrf_fusion(keyword, semantic)
        return self._rerank(query, fused, top_k=top_k)


def create_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [f"Source {i + 1}: {chunk.get('text', '')}" for i, chunk in enumerate(context_chunks)]
    )
    return f"""CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer only from the provided context.
2. If information is missing, say clearly what is missing.
3. Keep answer concise and factual.
"""


def generate_answer_hf(prompt: str, hf_model: str = GENERATOR_MODEL):
    from huggingface_hub import InferenceClient

    hf_token = os.getenv("HF_TOKEN")
    candidate_models = [hf_model, "google/flan-t5-large", "bigscience/bloom-560m"]
    last_error = None

    if hf_token and hf_token != "your_hf_token_here":
        client = InferenceClient(provider="hf-inference", api_key=hf_token)
        for model_name in candidate_models:
            start = time.time()
            try:
                out = client.text_generation(prompt, model=model_name, max_new_tokens=300, temperature=0.2)
                return out, time.time() - start
            except Exception as e:
                last_error = repr(e)

    global _local_gen_pipe
    if "_local_gen_pipe" not in globals() or _local_gen_pipe is None:
        _local_gen_pipe = pipeline("text-generation", model="distilgpt2")

    local_prompt = prompt
    try:
        tok = _local_gen_pipe.tokenizer
        max_positions = int(getattr(_local_gen_pipe.model.config, "n_positions", 1024))
        max_input_tokens = max(64, max_positions - 120)
        token_ids = tok.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_input_tokens:
            token_ids = token_ids[-max_input_tokens:]
            local_prompt = tok.decode(token_ids, skip_special_tokens=True)
    except Exception:
        pass

    start = time.time()
    out = _local_gen_pipe(local_prompt, max_new_tokens=120, do_sample=False)
    latency = time.time() - start
    if isinstance(out, list) and out:
        if "generated_text" in out[0]:
            return out[0]["generated_text"], latency
        return str(out[0]), latency

    if last_error:
        return f"Fallback output unavailable. Remote error: {last_error}", latency
    return str(out), latency


def call_hf_judge(prompt: str, model: str = JUDGE_MODEL) -> str:
    from huggingface_hub import InferenceClient

    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "your_hf_token_here":
        client = InferenceClient(provider="hf-inference", api_key=hf_token)
        candidate_models = [
            model,
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/flan-t5-large",
        ]
        for model_name in candidate_models:
            try:
                return client.text_generation(prompt, model=model_name, max_new_tokens=220, temperature=0.0)
            except Exception:
                continue

    global _local_judge_pipe
    if "_local_judge_pipe" not in globals() or _local_judge_pipe is None:
        _local_judge_pipe = pipeline("text-generation", model="distilgpt2")

    local_prompt = prompt
    try:
        tok = _local_judge_pipe.tokenizer
        max_positions = int(getattr(_local_judge_pipe.model.config, "n_positions", 1024))
        max_input_tokens = max(64, max_positions - 80)
        token_ids = tok.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_input_tokens:
            token_ids = token_ids[-max_input_tokens:]
            local_prompt = tok.decode(token_ids, skip_special_tokens=True)
    except Exception:
        pass

    out = _local_judge_pipe(local_prompt, max_new_tokens=80, do_sample=False)
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)


def extract_claims(answer_text: str) -> List[str]:
    prompt = f"""Extract atomic factual claims from the answer.
Return only a JSON array of short claims.
Answer: {answer_text}"""
    out = call_hf_judge(prompt)
    try:
        arr_match = re.search(r"\[.*\]", out, re.DOTALL)
        if arr_match:
            parsed = eval(arr_match.group(0))
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()][:8]
    except Exception:
        pass
    lines = [line.strip("- ").strip() for line in out.split("\n") if line.strip()]
    return [line for line in lines if len(line) > 5][:8]


def verify_claims_against_context(claims: List[str], context_text: str):
    verdicts = []
    for claim in claims:
        prompt = (
            f"Context:\n{context_text}\n\nClaim: {claim}\n\n"
            "Is this claim supported by context? Reply only with SUPPORTED or UNSUPPORTED."
        )
        out = call_hf_judge(prompt).upper()
        supported = "SUPPORTED" in out and "UNSUPPORTED" not in out
        verdicts.append({"claim": claim, "supported": supported})
    return verdicts


def faithfulness_score(answer_text: str, retrieved_chunks: List[Dict[str, Any]]):
    context_text = "\n\n".join([c.get("text", "") for c in retrieved_chunks])
    claims = extract_claims(answer_text)
    if not claims:
        return 0.0
    verdicts = verify_claims_against_context(claims, context_text)
    return float(sum(v["supported"] for v in verdicts) / len(verdicts))


def relevancy_score(original_query: str, answer_text: str, embedding_model: SentenceTransformer):
    prompt = (
        "Generate 3 alternative user questions that would have answer below. "
        f"Return only one question per line.\n\nAnswer:\n{answer_text}"
    )
    out = call_hf_judge(prompt)
    alt_qs = [line.strip(" -").strip() for line in out.split("\n") if line.strip()][:3]
    if not alt_qs:
        return 0.0

    q_vec = embedding_model.encode(original_query)
    sims = []
    for q in alt_qs:
        q2 = embedding_model.encode(q)
        sims.append(float(cosine_similarity([q_vec], [q2])[0][0]))
    return float(np.mean(sims))


STATE: Dict[str, Any] = {"ready": False, "retriever": None, "chunks": []}


def ensure_pipeline_ready() -> None:
    if STATE["ready"]:
        return

    default_corpus = Path(__file__).with_name("synthetic_knowledge_items.csv")
    corpus_path = os.getenv("CORPUS_PATH", str(default_corpus))

    mongo_collection = get_mongo_collection()
    chunks = load_chunks_from_mongodb(mongo_collection) if mongo_collection is not None else []

    if not chunks:
        docs = read_corpus_documents(corpus_path)
        if not docs:
            raise ValueError(f"No documents found at CORPUS_PATH={corpus_path}")
        chunks = build_chunks(docs)
        upsert_chunks_to_mongodb(mongo_collection, chunks)

    retriever = HybridRetriever()
    retriever.set_corpus(chunks)
    retriever.try_init_pinecone()
    retriever.upsert_to_pinecone(chunks)

    STATE["retriever"] = retriever
    STATE["chunks"] = chunks
    STATE["ready"] = True


def _format_context(chunks: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not chunks:
        return "No context chunks returned."

    lines = []
    for i, chunk in enumerate(chunks[:max_items], start=1):
        text = str(chunk.get("text", "")).strip()
        source = str(chunk.get("source", "unknown"))
        preview = (text[:350] + "...") if len(text) > 350 else text
        lines.append(f"[{i}] Source: {source}\n{preview}")
    return "\n\n".join(lines)


def run_rag(query: str):
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "", "", ""

    try:
        ensure_pipeline_ready()
        retriever: HybridRetriever = STATE["retriever"]

        retrieved = retriever.retrieve_hybrid(query, top_k=5)
        prompt = create_rag_prompt(query, retrieved)
        answer, _ = generate_answer_hf(prompt)

        faith = faithfulness_score(answer, retrieved)
        relev = relevancy_score(query, answer, retriever.embedding_model)

        return answer, _format_context(retrieved), f"{faith:.3f}", f"{relev:.3f}"
    except Exception as e:
        msg = f"Pipeline error: {repr(e)}"
        return msg, "", "N/A", "N/A"


with gr.Blocks(title="RAG Assignment 3") as demo:
    gr.Markdown("# RAG-based Question Answering System")
    gr.Markdown(
        "Set environment variables: HF_TOKEN, GENERATOR_MODEL, JUDGE_MODEL, MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION, PINECONE_API_KEY, PINECONE_ENVIRONMENT, CORPUS_PATH"
    )

    query_input = gr.Textbox(label="Ask a question", lines=2, placeholder="Type your question here...")
    submit_btn = gr.Button("Generate Answer", variant="primary")

    answer_output = gr.Textbox(label="Generated Answer", lines=8)
    context_output = gr.Textbox(label="Retrieved Context (Top Chunks)", lines=10)
    faithfulness_output = gr.Textbox(label="Faithfulness Score")
    relevancy_output = gr.Textbox(label="Relevancy Score")

    submit_btn.click(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output, context_output, faithfulness_output, relevancy_output],
    )
    query_input.submit(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output, context_output, faithfulness_output, relevancy_output],
    )


if __name__ == "__main__":
    print("Starting RAG QA system...")
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)