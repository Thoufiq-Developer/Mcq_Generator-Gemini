import os
import json
import asyncio
import time
from typing import List, Optional, Any
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEN_EMBED_MODEL = os.getenv("GEN_EMBED_MODEL", "models/embedding-001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gen_model = genai.GenerativeModel(GEMINI_MODEL)
    else:
        gen_model = None
except Exception:
    genai = None
    gen_model = None

try:
    from pypdf import PdfReader as PypdfReader
    PDF_READER = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader as PypdfReader
        PDF_READER = "pypdf2"
    except Exception:
        PypdfReader = None
        PDF_READER = None

import requests

st.set_page_config(page_title="Modern Legal MCQ / RAG (Gemini)", layout="wide")
st.title("Modern Legal MCQ / RAG — Gemini (modern deps)")

def extract_text_from_pdf_fileobj(fileobj) -> str:
    if PypdfReader is None:
        try:
            raw = fileobj.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        except Exception:
            return ""
    try:
        fileobj.seek(0)
    except Exception:
        pass
    try:
        reader = PypdfReader(fileobj)
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except Exception:
        try:
            fileobj.seek(0)
            raw = fileobj.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        except Exception:
            return ""

def safe_json_parse(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = text[start:end+1]
        try:
            return json.loads(fragment)
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

class GeminiClient:
    def __init__(self, embed_model: str = GEN_EMBED_MODEL, gen: Optional[Any] = gen_model):
        self.embed_model = embed_model
        self.gen = gen
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if genai is None:
            raise RuntimeError("Embeddings not available: google.generativeai not configured")
        results = genai.embeddings.create(model=self.embed_model, input=texts)
        if isinstance(results, dict) and "data" in results:
            return [item["embedding"] for item in results["data"]]
        # fallback shape
        out = []
        for item in getattr(results, "data", []):
            emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
            out.append(emb)
        return out
    def generate(self, prompt: str, timeout_s: int = 60) -> str:
        if self.gen is None:
            raise RuntimeError("Generative model not configured")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            resp = loop.run_until_complete(self.gen.generate_content_async(prompt, timeout=timeout_s))
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
            return text
        except Exception:
            resp = self.gen.generate_content(prompt)
            return getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))

@st.cache_resource
def get_gemini_client() -> GeminiClient:
    return GeminiClient()

gemini = get_gemini_client()

def build_mcq_prompt(doc_text: str, num: int = 5, difficulty: str = "medium") -> str:
    doc_preview = doc_text[:20000]
    prompt = f"""
You are an expert instructor. Create {num} multiple-choice questions (MCQs) from the document below.
Return valid JSON ONLY with top-level key "quiz" that is a list of questions.
Each question object must have:
- question: string
- choices: array of 4 strings
- answer: integer index (0-3)
- explanation: optional string

Document:
\"\"\"{doc_preview}\"\"\"

Tone/difficulty: {difficulty}
Do not emit any commentary outside the JSON.
"""
    return prompt

st.sidebar.header("Input / Settings")
uploaded = st.sidebar.file_uploader("Upload PDF or text file (pdf, txt, md, json)", type=["pdf","txt","md","json"])
manual_text = st.sidebar.text_area("Or paste text (optional)", height=160)
num_q = st.sidebar.number_input("Number of MCQs", min_value=1, max_value=50, value=5)
difficulty = st.sidebar.selectbox("Difficulty", ["easy","medium","hard"], index=1)
generate_btn = st.sidebar.button("Generate MCQs")

if generate_btn:
    doc_text = ""
    if uploaded is not None:
        doc_text = extract_text_from_pdf_fileobj(uploaded)
    if not doc_text and manual_text:
        doc_text = manual_text
    if not doc_text:
        st.sidebar.error("Provide a PDF or paste text first.")
    else:
        prompt = build_mcq_prompt(doc_text, num=num_q, difficulty=difficulty)
        with st.spinner("Requesting Gemini..."):
            try:
                out = gemini.generate(prompt)
            except Exception as e:
                st.error("Generation error: " + str(e))
                out = ""
            parsed = safe_json_parse(out)
            if not parsed or "quiz" not in parsed:
                st.error("Could not parse model JSON output. Raw output shown below.")
                st.code(out[:20000])
            else:
                quiz = parsed["quiz"]
                for i, q in enumerate(quiz, start=1):
                    st.markdown(f"### Q{i}. {q.get('question','')}")
                    choices = q.get("choices", [])
                    for idx, c in enumerate(choices):
                        mark = "✅" if idx == q.get("answer", -1) else "○"
                        st.write(f"{mark} {idx}. {c}")
                    if q.get("explanation"):
                        st.info(q["explanation"])
