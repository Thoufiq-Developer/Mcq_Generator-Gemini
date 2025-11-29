import os
import json
import asyncio
from typing import Any, List
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEN_EMBED_MODEL = os.getenv("GEN_EMBED_MODEL", "models/embedding-001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Gemini import/config
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

# PDF reader (modern)
try:
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None


# ---- Utility functions ---- #

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF or fallback to raw decoding."""
    if PdfReader is None:
        try:
            raw = uploaded_file.read()
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def safe_json_parse(text: str):
    """Extract JSON-like block safely."""
    if not text:
        return None

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    try:
        return json.loads(text)
    except Exception:
        return None


class GeminiClient:
    """Wrapper for embeddings & generation."""

    def __init__(self, embed_model: str = GEN_EMBED_MODEL, gen_model_ref=None):
        self.embed_model = embed_model
        self.gen = gen_model_ref

    def generate(self, prompt: str) -> str:
        if self.gen is None:
            raise RuntimeError("Gemini not configured")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            resp = loop.run_until_complete(self.gen.generate_content_async(prompt))
            return getattr(resp, "text", None) or str(resp)
        except Exception:
            resp = self.gen.generate_content(prompt)
            return getattr(resp, "text", None) or str(resp)


@st.cache_resource
def get_gemini():
    return GeminiClient(gen_model_ref=gen_model)


# ---- Modern UI Layout ---- #

st.set_page_config(
    page_title="MCQ Generator",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 16px;
            margin-top: -10px;
            color: #555;
        }
        .section-header {
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .question-box {
            padding: 18px;
            border-radius: 6px;
            background: #f9f9f9;
            margin-bottom: 20px;
            border: 1px solid #dedede;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>MCQ Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a document or paste text and generate high-quality MCQs instantly.</div>",
    unsafe_allow_html=True
)


# ---- Sidebar ---- #

st.sidebar.header("Input Options")

uploaded = st.sidebar.file_uploader(
    "Upload PDF or Text File",
    type=["pdf", "txt", "md", "json"]
)

manual_text = st.sidebar.text_area(
    "Or paste text here",
    placeholder="Paste content here...",
    height=160
)

num_q = st.sidebar.number_input(
    "Number of MCQs",
    min_value=1,
    max_value=50,
    value=5
)

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["easy", "medium", "hard"],
    index=1
)

generate_btn = st.sidebar.button("Generate")


# ---- Main Logic ---- #

if generate_btn:
    gemini = get_gemini()

    # Collect text
    source_text = ""

    if uploaded is not None:
        source_text = extract_pdf_text(uploaded)

    if not source_text and manual_text:
        source_text = manual_text

    if not source_text:
        st.error("Please upload a document or paste text.")
        st.stop()

    # Build prompt
    prompt = f"""
Generate {num_q} multiple-choice questions (MCQs) from the given content.
Return valid JSON only in the following format:

{{
  "quiz": [
    {{
      "question": "...",
      "choices": ["A", "B", "C", "D"],
      "answer": 1,
      "explanation": "..."
    }}
  ]
}}

Content:
\"\"\"{source_text[:15000]}\"\"\"

Difficulty: {difficulty}
Do not include any text outside JSON.
"""

    with st.spinner("Generating MCQs..."):
        try:
            output = gemini.generate(prompt)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    parsed = safe_json_parse(output)

    if not parsed or "quiz" not in parsed:
        st.error("Model output was not valid JSON. See raw output below.")
        st.code(output)
        st.stop()

    quiz = parsed["quiz"]

    st.markdown("<div class='section-header'>Generated MCQs</div>", unsafe_allow_html=True)

    # Display MCQs
    for idx, q in enumerate(quiz, start=1):
        with st.container():
            st.markdown(f"<div class='question-box'>", unsafe_allow_html=True)
            st.markdown(f"**Q{idx}. {q.get('question', '')}**")

            choices = q.get("choices", [])
            correct = q.get("answer", -1)

            for i, choice in enumerate(choices):
                prefix = "**(Correct Answer)**" if i == correct else ""
                st.write(f"- {choice} {prefix}")

            if q.get("explanation"):
                st.write(f"*Explanation:* {q['explanation']}")

            st.markdown("</div>", unsafe_allow_html=True)
