import os
import json
import asyncio
import traceback
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

load_dotenv()
API_KEY = os.getenv("google_api_key") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
if API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        gen_model = genai.GenerativeModel(MODEL_NAME)
    except Exception:
        genai = None
        gen_model = None
else:
    genai = None
    gen_model = None

def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(file_bytes)
        pages = []
        for p in reader.pages:
            text = p.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""

def read_uploaded_file(uploaded):
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded)
    try:
        raw = uploaded.read()
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="ignore")
            except Exception:
                return str(raw)
        return str(raw)
    except Exception:
        return ""

def call_genai_for_mcqs(text, number=5, subject="", tone="simple"):
    if gen_model is None:
        raise RuntimeError("Generative model not configured. Set google_api_key secret.")
    prompt = f"""
Generate {number} MCQs in JSON format from the following document. Output MUST be valid JSON with top-level key "quiz" which is a list of objects with keys: question, choices (list), answer (index of correct choice starting 0), explanation (optional).
Document:
\"\"\"{text}\"\"\"
Subject: {subject}
Tone: {tone}
Return only JSON.
"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        try:
            resp = loop.run_until_complete(gen_model.generate_content_async(prompt))
            text_out = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
        except Exception:
            resp = gen_model.generate_content(prompt)
            text_out = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else str(resp))
        start = text_out.find("{")
        if start != -1:
            text_out = text_out[start:]
        data = json.loads(text_out)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to generate or parse JSON from model response: {e}\nRaw response: {text_out[:1000]}")

st.set_page_config(page_title="MCQ Generator", layout="wide")
st.title("MCQ Generator — Streamlit + Gemini (fallback safe)")

with st.form("mcq"):
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf","txt","md"])
    num = st.number_input("Number of MCQs", value=5, min_value=1, max_value=50)
    subject = st.text_input("Subject (optional)")
    tone = st.selectbox("Tone", ["simple","intermediate","complex"])
    submit = st.form_submit_button("Generate MCQs")
    if submit:
        if uploaded is None:
            st.error("Upload a PDF or text file first.")
        else:
            with st.spinner("Extracting text..."):
                try:
                    doc_text = read_uploaded_file(uploaded)
                except Exception as e:
                    st.error("Cannot read uploaded file.")
                    st.exception(e)
                    doc_text = ""
            if not doc_text.strip():
                st.error("No text found in the file.")
            else:
                with st.spinner("Generating MCQs..."):
                    try:
                        result = call_genai_for_mcqs(doc_text, number=int(num), subject=subject, tone=tone)
                        quiz = result.get("quiz") if isinstance(result, dict) else None
                        if not quiz:
                            st.error("Model did not return a quiz key or returned empty quiz.")
                            st.code(json.dumps(result, indent=2))
                        else:
                            rows = []
                            for i, q in enumerate(quiz):
                                question = q.get("question","")
                                choices = q.get("choices",[])[:]
                                ans_index = q.get("answer",0)
                                explanation = q.get("explanation","")
                                rows.append({
                                    "No": i+1,
                                    "Question": question,
                                    "Choices": "\n".join([f"{idx}. {c}" for idx,c in enumerate(choices)]),
                                    "Answer index": ans_index,
                                    "Explanation": explanation
                                })
                            df = pd.DataFrame(rows)
                            st.success("MCQs generated")
                            st.table(df)
                    except Exception as e:
                        st.error("Generation failed. See details below.")
                        st.exception(e)
