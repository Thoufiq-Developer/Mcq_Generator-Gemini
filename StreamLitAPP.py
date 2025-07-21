import os
import json
import traceback
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.mcqgenerator import connection_chain,estimate_gemini_token_usage
from src.mcqgenerator.logger import logging

load_dotenv()
key=os.getenv('google_api_key')

st.title('🔭🔭 MCQ Creator application with langchain') 
st.write("This application generates MCQs from the provided text or PDF file using LangChain and Gemini's flash 2.5 model.")


with st.form("user input"):
    user_input=st.file_uploader("upload the file here")
    number=st.number_input("Enter the number of MCQs to generate:", min_value=1, max_value=50)
    subject=st.text_input("Insert Subject ", max_chars=50)
    tone=st.text_input("complexity level of the quiz :simple, intermediate, complex")
    button=st.form_submit_button("Generate MCQs")

    if button and user_input is not None and number and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(user_input)
                with open("Response.json", "r") as f:
                    RESPONSE_JSON = json.load(f)

                with get_openai_callback() as cb:
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