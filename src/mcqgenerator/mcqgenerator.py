import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

load_dotenv()
key = os.getenv("google_api_key")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=key)

TEMPLATE = """
TEXT: {text}
You are an expert MCQ maker. Based on the above text, create a quiz of {number} multiple choice questions
for {subject} students in a {tone} tone.

Ensure:
- No repeated questions
- Questions strictly based on the text
- Output formatted like RESPONSE_JSON below

### RESPONSE_JSON
{response_json}
"""

TEMPLATE2 = """
You are a grammar expert and quiz evaluator. Given the following MCQs for {subject} students:

Evaluate:
- Complexity (max 50 words)
- Suggest any changes if the difficulty doesn't match student level

MCQs:
{quiz}

Your analysis:
"""

prompt_template = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_eval_template = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

chain = LLMChain(llm=llm, prompt=prompt_template, output_key="quiz", verbose=False)
eval_chain = LLMChain(llm=llm, prompt=quiz_eval_template, output_key="review", verbose=False)

connection_chain = SequentialChain(
    chains=[chain, eval_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=False
)

def estimate_gemini_token_usage(prompt_inputs: dict, model_outputs: dict = None) -> int:
    input_tokens = sum(len(str(value).split()) for value in prompt_inputs.values())
    output_tokens = sum(len(str(value).split()) for value in model_outputs.values()) if model_outputs else 0
    return input_tokens + output_tokens
