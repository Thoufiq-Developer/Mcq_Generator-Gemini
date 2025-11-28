import os
import json
import asyncio
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import requests

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEN_EMBED_MODEL = os.getenv("GEN_EMBED_MODEL", "models/embedding-001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gen_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        genai = None
        gen_model = None
else:
    genai = None
    gen_model = None

def extract_text_from_pdf_bytes(f):
    try:
        reader = PdfReader(f)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception:
        try:
            raw = f.read().decode("utf-8", errors="ignore")
            return raw
        except Exception:
            return ""

def extract_text_from_json_bytes(f):
    try:
        raw = f.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        data = json.loads(raw)
    except Exception:
        return ""
    texts = []
    def walk(o):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)
    walk(data)
    return "\n".join([t for t in texts if len(t.strip())>0])

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

class GeminiEmb:
    def __init__(self, model_name=GEN_EMBED_MODEL, batch_size=16):
        self.model_name = model_name
        self.batch_size = batch_size
    def _call(self, inputs):
        if genai is None:
            raise RuntimeError("Gemini embeddings unavailable; set GEMINI_API_KEY secret")
        resp = genai.embeddings.create(model=self.model_name, input=inputs)
        if isinstance(resp, dict) and "data" in resp:
            return [it.get("embedding") for it in resp["data"]]
        data = getattr(resp, "data", [])
        out = []
        for it in data:
            emb = getattr(it, "embedding", None) or (it.get("embedding") if isinstance(it, dict) else None)
            out.append(emb)
        return out
    def embed_documents(self, texts):
        if not texts:
            return []
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_emb = self._call(batch)
            embs.extend(batch_emb)
        return embs
    def embed_query(self, text):
        out = self._call([text])
        return out[0]

class SimpleVectorStore:
    def __init__(self):
        self.ids = []
        self.docs = []
        self.embeddings = None
    def add(self, ids, docs, embeddings):
        self.ids.extend(ids)
        self.docs.extend(docs)
        arr = np.array(embeddings, dtype=float)
        if self.embeddings is None:
            self.embeddings = arr
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
    def similarity_search(self, query_emb, k=3):
        if self.embeddings is None or len(self.docs)==0:
            return []
        q = np.array(query_emb, dtype=float)
        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q) + 1e-12)
        sims = (self.embeddings @ q) / norms
        idx = np.argsort(-sims)[:k]
        results = [{"id": self.ids[i], "doc": self.docs[i], "score": float(sims[i])} for i in idx]
        return results
    def clear(self):
        self.ids = []
        self.docs = []
        self.embeddings = None

@st.cache_resource
def get_embedder():
    return GeminiEmb()

embedder = get_embedder()
if "vstore" not in st.session_state:
    st.session_state.vstore = SimpleVectorStore()
    st.session_state.chunks_count = 0

st.set_page_config(page_title="Legal RAG (Gemini)", layout="wide")
st.title("Legal RAG — Gemini embeddings + in-memory store")

with st.sidebar:
    st.header("Index documents")
    uploaded = st.file_uploader("Upload PDF or JSON", type=["pdf","json","txt","md"])
    chunk_size = st.number_input("Chunk size", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, step=50)
    if st.button("Index file") and uploaded is not None:
        with st.spinner("Reading file..."):
            try:
                if uploaded.name.lower().endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(uploaded)
                elif uploaded.name.lower().endswith(".json"):
                    text = extract_text_from_json_bytes(uploaded)
                else:
                    raw = uploaded.read()
                    if isinstance(raw, bytes):
                        try:
                            text = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            text = str(raw)
                    else:
                        text = str(raw)
            except Exception:
                text = ""
        if not text.strip():
            st.error("No text found in file")
        else:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            with st.spinner("Creating embeddings..."):
                try:
                    embs = embedder.embed_documents(chunks)
                    ids = [f"{uploaded.name}_chunk_{i}" for i in range(len(chunks))]
                    st.session_state.vstore.add(ids, chunks, embs)
                    st.session_state.chunks_count = len(st.session_state.vstore.docs)
                    st.success(f"Indexed {len(chunks)} chunks (total {st.session_state.chunks_count})")
                except Exception as e:
                    st.error("Embedding failed")
                    st.exception(e)
    if st.button("Clear index"):
        st.session_state.vstore.clear()
        st.session_state.chunks_count = 0
        st.success("Index cleared")
    st.markdown("---")
    st.markdown("If you have a file named `Response.json` in the repo it will also be indexed automatically on startup.")
    st.markdown("Secrets (Settings → Secrets) example:")
    st.code('''GEMINI_API_KEY = "YOUR_KEY"\nGEN_EMBED_MODEL = "models/embedding-001"\nGEMINI_MODEL = "gemini-1.5-flash"''')

if os.path.exists("Response.json") and st.session_state.chunks_count==0:
    try:
        with open("Response.json","r",encoding="utf-8") as f:
            txt = ""
            try:
                data = json.load(f)
                def walk(o):
                    if isinstance(o, str):
                        return o
                    if isinstance(o, dict):
                        return " ".join([walk(v) for v in o.values()])
                    if isinstance(o, list):
                        return " ".join([walk(i) for i in o])
                    return ""
                txt = walk(data)
            except Exception:
                f.seek(0)
                txt = f.read()
            chunks = chunk_text(txt)
            try:
                embs = embedder.embed_documents(chunks)
                ids = [f"Response_json_chunk_{i}" for i in range(len(chunks))]
                st.session_state.vstore.add(ids, chunks, embs)
                st.session_state.chunks_count = len(st.session_state.vstore.docs)
            except Exception:
                pass
    except Exception:
        pass

st.subheader("Ask a question")
query = st.text_input("Enter your question")
if st.button("Search") and query.strip():
    if st.session_state.chunks_count==0:
        st.error("No indexed documents. Upload and index a file first.")
    else:
        with st.spinner("Embedding query and searching..."):
            try:
                q_emb = embedder.embed_query(query)
                hits = st.session_state.vstore.similarity_search(q_emb, k=3)
                context = "\n\n".join([h["doc"] for h in hits])
            except Exception as e:
                st.error("Search failed")
                st.exception(e)
                context = ""
        prompt = f"You are a legal assistant. This is not legal advice.\n\nDocument context:\n{context}\n\nQuestion:\n{query}"
        if gen_model is None:
            st.error("Generative model not configured. Set GEMINI_API_KEY in Secrets.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    try:
                        resp = loop.run_until_complete(gen_model.generate_content_async(prompt))
                        ans = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
                    except Exception:
                        resp = gen_model.generate_content(prompt)
                        ans = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
                    st.markdown("**Answer:**")
                    st.write(ans)
                    if hits:
                        st.markdown("**Sources (top matches):**")
                        for h in hits:
                            st.write(f"- score: {h['score']:.4f}  snippet: {h['doc'][:300].strip().replace('\\n',' ')}")
                except Exception as e:
                    st.error("Generation failed")
                    st.exception(e)
