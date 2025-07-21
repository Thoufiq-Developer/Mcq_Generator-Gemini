import os
import json
import pandas as pd 

from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.mcqgenerator.utils import read_file,get_table_data

load_dotenv()
key=os.getenv('google_api_key')

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.1,google_api_key=key)

TEMPLATE='''
TEXT:{text}
you are an expert MCQ maker . Given the above text , it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in{tone} tone.
make sure that no question is repeated and check all the questions to be conforming the text as well
make sure to format your response like RESPONSE_JSON below and use it like a guide. \
Ensure that to make {number} MCQs
### RESPONSE_JSON
{response_json}

'''

TEMPLATE2='''
you are an expert in english grammar and writer . Given a multiple choice quiz for {subject} students.\
you need to evaluate the complexity of the quiz and goive a complete analysis of the quiz . Only use at max 50 wordsfor complexity
if the quiz is not as per the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the studentsabilities
Quiz_MCQs:
{quiz}

check from an expert English Writer of the above quiz:
'''

prompt_template=PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

chain=LLMChain(llm=llm,prompt=prompt_template,output_key="quiz",verbose=False)
quiz_evaluation_template=PromptTemplate(input_variables=["subject","quiz"],template=TEMPLATE2)
quiz_evaluation_chain=LLMChain(llm=llm,prompt=quiz_evaluation_template,output_key="review",verbose=False)
connection_chain=SequentialChain(chains=[chain,quiz_evaluation_chain],input_variables=["text","number","subject","tone","response_json"],output_variables=["quiz","review"],verbose=False)

def estimate_gemini_token_usage(prompt_inputs: dict, model_outputs: dict = None) -> int:
    input_tokens = 0
    for key, value in prompt_inputs.items():
        if isinstance(value, str):
            input_tokens += len(value.split())
        elif isinstance(value, dict):
            input_tokens += len(json.dumps(value).split())
        else:
            input_tokens += len(str(value).split())
    output_tokens = 0
    if model_outputs:
        for key, value in model_outputs.items():
            if isinstance(value, str):
                
                output_tokens += len(value.split())
            elif isinstance(value, dict):
                output_tokens += len(json.dumps(value).split())
            else:
                output_tokens += len(str(value).split())
    return input_tokens + output_tokens


 
     


