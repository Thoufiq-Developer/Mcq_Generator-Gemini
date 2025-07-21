import os
import json
import traceback
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback


from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.mcqgenerator import connection_chain, estimate_gemini_token_usage
from src.mcqgenerator.logger import logging

# Load Gemini API key
load_dotenv()
key = os.getenv("google_api_key")

# UI Setup
st.title('📘 MCQ Creator App with Gemini & LangChain')
st.write("Upload a .txt or .pdf file, and get high-quality MCQs.")

# File Upload Form
with st.form("user_input"):
    user_input = st.file_uploader("Upload the file here (.txt or .pdf)", type=["txt", "pdf"])
    number = st.number_input("Number of MCQs:", min_value=1, max_value=50)
    subject = st.text_input("Subject:", max_chars=20)
    tone = st.text_input("Tone (e.g., simple, intermediate, complex):")
    button = st.form_submit_button("Generate MCQs")

    if button and user_input and subject and tone:
        with st.spinner("Generating MCQs..."):
            try:
                text = read_file(user_input)
                with open("Response.json", "r") as f:
                    RESPONSE_JSON = json.load(f)

                with  get_openai_callback() as cb:
                    response = connection_chain.invoke({
                        "text": text,
                        "number": number,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    })

            except Exception as e:
                st.error("Something went wrong.")
                traceback.print_exception(type(e), e, e.__traceback__)
            else:
                st.success("MCQs Generated Successfully!")
                st.text(f"Total Tokens Used: {cb.total_tokens}")
                if isinstance(response, dict):
                    quiz = response.get("quiz")
                    table_data = get_table_data(response)
                    if table_data:
                        df = pd.DataFrame(table_data)
                        df.index += 1
                        st.table(df)
                        st.text_area("Review:", value=response.get("review", "No review found."))
                    else:
                        st.error("No MCQs were parsed from the response.")
                else:
                    st.write(response)
