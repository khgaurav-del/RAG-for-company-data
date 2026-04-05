
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

import spacy
nlp = spacy.load('en_core_web_md')

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

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

try:
    from groq import Groq
except Exception:
    Groq = None


GENERATOR_MODEL = os.getenv("GROQ_GENERATOR_MODEL", os.getenv("GENERATOR_MODEL", "deepseek-r1-distill-qwen-14b"))
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"))
TRANSLATION_MODEL = os.getenv("GROQ_TRANSLATION_MODEL", os.getenv("TRANSLATION_MODEL", "llama-3.1-8b-instant"))
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "semantic").strip().lower() or "semantic"
CHUNK_CACHE_VERSION = os.getenv("CHUNK_CACHE_VERSION", "v6")
CHUNK_CACHE_DIR = Path(__file__).with_name("chunk_cache")
CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: Dict[str, Any] = {}

URDU_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


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

    def load_parquet_file(fp: Path) -> None:
        try:
            df = pd.read_parquet(fp)
        except ImportError as exc:
            raise ImportError("Reading parquet files requires 'pyarrow' or 'fastparquet'.") from exc

        if len(df.columns) > 1:
            # Matches notebook behavior: ignore likely index/id first column for parquet corpora.
            df = df.iloc[:, 1:].copy()

        text_cols = [
            c
            for c in df.columns
            if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
        ]
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
        elif suffix == ".parquet":
            load_parquet_file(path)
        return docs

    if path.is_dir():
        for pattern in ["*.txt", "*.md"]:
            for fp in path.rglob(pattern):
                load_text_file(fp)
        for fp in path.rglob("*.csv"):
            load_csv_file(fp)
        for fp in path.rglob("*.parquet"):
            load_parquet_file(fp)

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


def get_cached_embedding_model(model_name: str):
    cache_key = f"embedding::{model_name}"
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[cache_key] = model
    return model


def _corpus_signature(corpus_path: Path) -> str:
    import hashlib

    h = hashlib.sha1()
    corpus_path = Path(corpus_path)

    if corpus_path.is_file():
        st = corpus_path.stat()
        h.update(f"{corpus_path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8"))
        return h.hexdigest()[:12]

    if corpus_path.is_dir():
        supported = {".txt", ".md", ".csv", ".parquet"}
        for fp in sorted(corpus_path.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in supported:
                st = fp.stat()
                h.update(f"{fp.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8"))
        return h.hexdigest()[:12]

    h.update(str(corpus_path).encode("utf-8"))
    return h.hexdigest()[:12]


def get_chunk_cache_path(corpus_path: str, chunking_method: str = "semantic") -> Path:
    import hashlib

    corpus_path_obj = Path(corpus_path)
    corpus_name = corpus_path_obj.stem if corpus_path_obj.is_file() else corpus_path_obj.name
    corpus_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", corpus_name or "corpus").strip("_").lower()
    resolved = str(corpus_path_obj.resolve()) if corpus_path_obj.exists() else str(corpus_path_obj)
    source_sig = _corpus_signature(corpus_path_obj)
    chunk_cfg = f"{CHUNK_CACHE_VERSION}|{chunking_method}"
    digest = hashlib.sha1(f"{resolved}|{source_sig}|{chunk_cfg}".encode("utf-8")).hexdigest()[:12]
    return CHUNK_CACHE_DIR / f"{corpus_name}_{chunking_method}_{CHUNK_CACHE_VERSION}_{digest}.jsonl"


def save_chunks_to_cache(chunks: List[Dict[str, str]], cache_path: Path) -> None:
    import json

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_id = str(chunk.get("id", "")).strip()
            text = str(chunk.get("text", "")).strip()
            source = str(chunk.get("source", "unknown")).strip() or "unknown"
            if chunk_id and text:
                record = {"id": chunk_id, "text": text, "source": source}
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")


def load_chunks_from_cache(cache_path: Path) -> List[Dict[str, str]]:
    import json

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return []

    chunks: List[Dict[str, str]] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = str(record.get("id", "")).strip()
            text = str(record.get("text", "")).strip()
            source = str(record.get("source", "unknown")).strip() or "unknown"
            if chunk_id and text:
                chunks.append({"id": chunk_id, "text": text, "source": source})

    return chunks


def get_semantic_cache_path(cache_path: Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix(".semantic.npy")


def save_semantic_embeddings_to_cache(embedding_matrix, semantic_cache_path: Path) -> None:
    if embedding_matrix is None:
        return
    semantic_cache_path = Path(semantic_cache_path)
    semantic_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(semantic_cache_path, embedding_matrix)


def load_semantic_embeddings_from_cache(semantic_cache_path: Path, expected_rows: int | None = None):
    semantic_cache_path = Path(semantic_cache_path)
    if not semantic_cache_path.exists():
        return None

    try:
        matrix = np.load(semantic_cache_path, allow_pickle=False)
    except Exception:
        return None

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        return None

    if expected_rows is not None and matrix.shape[0] != int(expected_rows):
        return None

    return matrix


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
        self.embedding_model = get_cached_embedding_model(embedding_model_name)
        self.bm25_model = None
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.local_chunk_matrix = None
        self.pc_client = None
        self.pc_index = None
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-assignment3-index")

    def set_corpus(self, chunks: List[Dict[str, Any]], semantic_matrix=None) -> None:
        self.bm25_chunks = chunks
        if BM25Okapi is not None:
            tokenized = [c["text"].lower().split() for c in chunks]
            self.bm25_model = BM25Okapi(tokenized)
        if semantic_matrix is not None and len(semantic_matrix) == len(chunks):
            self.local_chunk_matrix = np.array(semantic_matrix)
        else:
            vectors = self.embedding_model.encode([c["text"] for c in chunks], show_progress_bar=False)
            self.local_chunk_matrix = np.array(vectors)

    def try_init_pinecone(self) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        region = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        if not (Pinecone and ServerlessSpec and api_key):
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


def _resolve_groq_api_key() -> str:
    return (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("Groq_API_KEY", "").strip()
    )


def get_cached_groq_client():
    client = _MODEL_CACHE.get("groq_client")
    if client is not None:
        return client

    api_key = _resolve_groq_api_key()
    if not api_key or Groq is None:
        return None

    try:
        client = Groq(api_key=api_key)
        _MODEL_CACHE["groq_client"] = client
        return client
    except Exception:
        return None


def _dedupe_models(*models):
    seen = set()
    ordered = []
    for m in models:
        name = str(m or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _groq_chat_completion(prompt: str, model_name: str, max_tokens: int = 450, temperature: float = 0.2) -> str:
    client = get_cached_groq_client()
    if client is None:
        raise RuntimeError("Groq client is not initialized. Set GROQ_API_KEY first.")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a concise RAG assistant. Answer using only the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response or not getattr(response, "choices", None):
        return ""
    msg = response.choices[0].message
    return (msg.content or "").strip()


def _clean_generated_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:1800]


def is_pure_urdu_text(text: str, min_urdu_ratio: float = 0.85) -> bool:
    text = str(text or "").strip()
    if not text:
        return False

    if LATIN_CHAR_RE.search(text):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    urdu_letters = [ch for ch in letters if URDU_CHAR_RE.fullmatch(ch)]
    ratio = len(urdu_letters) / max(1, len(letters))
    return ratio >= min_urdu_ratio


def _translate_text(text: str, source_lang: str, target_lang: str, enforce_pure_urdu: bool = False) -> str:
    text = str(text or "").strip()
    if not text:
        return text

    model_name = os.getenv("GROQ_TRANSLATION_MODEL", "").strip() or TRANSLATION_MODEL or GENERATOR_MODEL
    purity_rule = ""
    if enforce_pure_urdu:
        purity_rule = "- Output must be in Urdu script only. Do not use English words or Roman Urdu.\\n"

    prompt = (
        f"Translate the following {source_lang} text to {target_lang}.\\n"
        "Rules:\\n"
        "- Preserve meaning faithfully and keep tone natural.\\n"
        "- Keep names, numbers, and dates unchanged when possible.\\n"
        f"{purity_rule}"
        "- Return only the translation text, without notes or quotes.\\n\\n"
        f"Text:\\n{text}"
    )

    try:
        translated = _groq_chat_completion(prompt, model_name=model_name, max_tokens=600, temperature=0.0)
        translated = str(translated or "").strip()
        if translated:
            return translated
    except Exception:
        pass

    return text


def translate_urdu_to_english(text: str) -> str:
    return _translate_text(text, source_lang="Urdu", target_lang="English")


def translate_english_to_pure_urdu(text: str) -> str:
    return _translate_text(
        text,
        source_lang="English",
        target_lang="Urdu",
        enforce_pure_urdu=True,
    )


def _extractive_fallback_answer(prompt: str, max_points: int = 6) -> str:
    context_match = re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:", prompt, re.DOTALL)
    if not context_match:
        return "Not found in provided context."

    context_block = context_match.group(1).strip()
    if not context_block:
        return "Not found in provided context."

    lines = [line.strip() for line in context_block.split("\n") if line.strip()]
    picked = []
    for line in lines:
        if line.lower().startswith("source "):
            source_part = line.split(":", 1)
            if len(source_part) == 2 and source_part[1].strip():
                picked.append(f"- {source_part[1].strip()}")
        if len(picked) >= max_points:
            break

    if not picked:
        return "Not found in provided context."
    return "\n".join(picked)


def generate_answer_hf(prompt: str, hf_model: str = GENERATOR_MODEL):
    generator_model = hf_model or os.getenv("GROQ_GENERATOR_MODEL") or GENERATOR_MODEL
    candidate_models = _dedupe_models(generator_model, "llama-3.1-8b-instant", "llama3-8b-8192")
    last_error = None

    for model_name in candidate_models:
        start = time.time()
        try:
            out = _groq_chat_completion(prompt, model_name=model_name, max_tokens=450, temperature=0.2)
            return _clean_generated_answer(out), time.time() - start
        except Exception as e:
            last_error = repr(e)

    extractive_answer = _extractive_fallback_answer(prompt, max_points=6)
    if extractive_answer:
        return extractive_answer, 0.0
    return f"Groq generation failed: {last_error}", 0.0


def call_hf_judge(prompt: str, model: str = JUDGE_MODEL) -> str:
    judge_model = model or os.getenv("GROQ_JUDGE_MODEL") or JUDGE_MODEL
    candidate_models = _dedupe_models(
        judge_model,
        os.getenv("GROQ_JUDGE_MODEL"),
        os.getenv("GROQ_GENERATOR_MODEL"),
        "llama-3.1-8b-instant",
    )
    last_error = None

    for model_name in candidate_models:
        try:
            return _groq_chat_completion(prompt[:4000], model_name=model_name, max_tokens=180, temperature=0.0)
        except Exception as e:
            last_error = repr(e)

    return f"Groq judge failed: {last_error}"


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


STATE: Dict[str, Any] = {"ready": False, "retriever": None, "chunks": [], "docs": []}


def ensure_pipeline_ready() -> None:
    if STATE["ready"]:
        return

    app_dir = Path(__file__).resolve().parent
    candidate_defaults = [
        app_dir / "Mental_Health_" / "support_1000.parquet",
        app_dir / "synthetic_knowledge_items.csv",
    ]
    default_corpus = next((p for p in candidate_defaults if p.exists()), candidate_defaults[-1])
    corpus_path = (os.getenv("CORPUS_PATH", str(default_corpus)) or str(default_corpus)).strip()

    force_rechunk = str(os.getenv("FORCE_RECHUNK", "false")).strip().lower() in {"1", "true", "yes", "y"}
    load_docs_on_cache_hit = str(os.getenv("LOAD_DOCS_ON_CACHE_HIT", "false")).strip().lower() in {"1", "true", "yes", "y"}
    upsert_on_cache_hit = str(os.getenv("UPSERT_ON_CACHE_HIT", "false")).strip().lower() in {"1", "true", "yes", "y"}

    cache_path = get_chunk_cache_path(corpus_path, chunking_method=CHUNKING_METHOD)
    semantic_cache_path = get_semantic_cache_path(cache_path)

    chunks: List[Dict[str, str]] = []
    docs: List[Dict[str, str]] = []
    chunk_cache_hit = False

    if not force_rechunk:
        chunks = load_chunks_from_cache(cache_path)
        chunk_cache_hit = len(chunks) > 0
        if chunk_cache_hit:
            print(f"Loaded {len(chunks)} chunks from cache: {cache_path}")
            print("Reusing cached chunks. Skipping chunking step.")

    mongo_collection = get_mongo_collection()
    if not chunks and mongo_collection is not None:
        chunks = load_chunks_from_mongodb(mongo_collection)
        if chunks:
            chunk_cache_hit = True
            save_chunks_to_cache(chunks, cache_path)
            print(f"Loaded {len(chunks)} chunks from MongoDB and saved local cache: {cache_path}")

    if not chunks:
        docs = read_corpus_documents(corpus_path)
        if not docs:
            raise ValueError(f"No documents found at CORPUS_PATH={corpus_path}")
        chunks = build_chunks(docs)
        save_chunks_to_cache(chunks, cache_path)
        upsert_chunks_to_mongodb(mongo_collection, chunks)
        print(f"Chunked corpus and saved {len(chunks)} chunks to cache: {cache_path}")
    elif load_docs_on_cache_hit:
        docs = read_corpus_documents(corpus_path)
    else:
        print("Skipping corpus read on cache hit for faster startup.")

    retriever = HybridRetriever()

    semantic_matrix = None
    if not force_rechunk:
        semantic_matrix = load_semantic_embeddings_from_cache(semantic_cache_path, expected_rows=len(chunks))
        if semantic_matrix is not None:
            print(f"Loaded semantic embedding matrix from cache: {semantic_cache_path.name}")

    retriever.set_corpus(chunks, semantic_matrix=semantic_matrix)

    if semantic_matrix is None and retriever.local_chunk_matrix is not None:
        save_semantic_embeddings_to_cache(retriever.local_chunk_matrix, semantic_cache_path)
        print(f"Saved semantic embedding matrix cache: {semantic_cache_path.name}")

    retriever.try_init_pinecone()
    should_upsert = (not chunk_cache_hit) or bool(upsert_on_cache_hit)
    if should_upsert:
        retriever.upsert_to_pinecone(chunks)
    else:
        print("Skipping Pinecone upsert on cache hit for faster startup.")

    STATE["retriever"] = retriever
    STATE["chunks"] = chunks
    STATE["docs"] = docs
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

        urdu_query = is_pure_urdu_text(query)
        rag_query = translate_urdu_to_english(query) if urdu_query else query

        retrieved = retriever.retrieve_hybrid(rag_query, top_k=5)
        prompt = create_rag_prompt(rag_query, retrieved)
        english_answer, _ = generate_answer_hf(prompt)

        answer = translate_english_to_pure_urdu(english_answer) if urdu_query else english_answer

        faith = faithfulness_score(english_answer, retrieved)
        relev = relevancy_score(rag_query, english_answer, retriever.embedding_model)

        return answer, _format_context(retrieved), f"{faith:.3f}", f"{relev:.3f}"
    except Exception as e:
        msg = f"Pipeline error: {repr(e)}"
        return msg, "", "N/A", "N/A"


with gr.Blocks(title="RAG Assignment 3") as demo:
    gr.Markdown("# Personal psychiatric assistant (RAG QA System)")

    query_input = gr.Textbox(label="Ask a question", lines=2, placeholder="Type your question here...")
    submit_btn = gr.Button("Ask RAG", variant="primary")

    answer_output = gr.Textbox(label="Generated Answer", lines=8)

    submit_btn.click(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output],
    )
    query_input.submit(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output],
    )


if __name__ == "__main__":
    print("Starting RAG QA system...")
    
    # Hugging Face uses port 7860 by default
    port = int(os.getenv("PORT", 7860))
    
    # Force host to 0.0.0.0 so the Space can be accessed externally
    host = "0.0.0.0" 
    
    print(f"Server is starting on {host}:{port}")

    # Launch Gradio
    demo.launch(
        server_name=host, 
        server_port=port, 
        share=False  # Keep share False on Hugging Face
    )