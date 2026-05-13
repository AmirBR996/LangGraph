from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = "sequential_App"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model_name="llama-3.3-70b-versatile")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    "tags" : ['llm_app' , 'report_generation' , 'summarization'],
    'metadata' : {"model1" : "grok llama-3.3-70b-versatile"}
}

result = chain.invoke({'topic': 'Unemployment in Nepal'} , config= config)

print(result)
