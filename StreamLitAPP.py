import os
import json
import asyncio
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
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

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        texts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                texts.append(t)
        return "\n\n".join(texts)
    except Exception:
        try:
            uploaded_file.seek(0)
            b = uploaded_file.read()
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="ignore")
            return str(b)
        except Exception:
            return ""

def attempt_parse_json(s):
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return None

st.set_page_config(page_title="Simple MCQ Generator", layout="wide")
st.title("Simple MCQ Generator — Gemini")

st.markdown("Upload a PDF or paste text. Gemini will generate MCQs in JSON and the app will show them.")

with st.sidebar:
    uploaded = st.file_uploader("Upload PDF (or text file)", type=["pdf","txt","md","json"])
    manual_text = st.text_area("Or paste text here (optional)", height=120)
    num_q = st.number_input("Number of MCQs", min_value=1, max_value=50, value=5)
    difficulty = st.selectbox("Difficulty / Tone", ["simple","medium","hard"], index=1)
    submit = st.button("Generate MCQs")

if submit:
    if gen_model is None:
        st.error("Gemini not configured. Set GEMINI_API_KEY secret and wait ~1 minute.")
    else:
        doc_text = ""
        if uploaded is not None:
            doc_text = extract_text_from_pdf(uploaded)
        if not doc_text and manual_text:
            doc_text = manual_text
        if not doc_text:
            st.error("No document text found. Upload a PDF or paste text.")
        else:
            prompt = f"""
Generate {int(num_q)} multiple choice questions (MCQs) from the document below.
Output must be valid JSON only. The JSON top-level object must be:
{{ "quiz": [ {{ "question": "...", "choices": ["...","...","...","..."], "answer": 0, "explanation": "..." } , ... ] }}

Document:
\"\"\"{doc_text[:15000]}\"\"\"

Each question must have exactly 4 choices. "answer" is the index (0-3) of the correct choice.
Do not include any commentary outside the JSON.
"""
            with st.spinner("Calling Gemini..."):
                try:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    try:
                        resp = loop.run_until_complete(gen_model.generate_content_async(prompt))
                        out = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
                    except Exception:
                        resp = gen_model.generate_content(prompt)
                        out = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
                    parsed = attempt_parse_json(out)
                    if parsed is None or "quiz" not in parsed:
                        st.error("Model response could not be parsed as expected. Raw output shown below.")
                        st.code(out)
                    else:
                        quiz = parsed.get("quiz", [])
                        for i, q in enumerate(quiz, 1):
                            st.markdown(f"### Q{i}. {q.get('question','')}")
                            choices = q.get("choices", [])
                            for idx, ch in enumerate(choices):
                                prefix = "✅" if idx == q.get("answer", -1) else "○"
                                st.write(f"{prefix} {idx}. {ch}")
                            if q.get("explanation"):
                                st.info(f"Explanation: {q.get('explanation')}")
                except Exception as e:
                    st.error("Error calling Gemini or parsing response.")
                    st.exception(e)
