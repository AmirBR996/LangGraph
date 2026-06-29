from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# state

class State(TypedDict):
    question: str
    answer_eng: str
    input_text: str
    translated_text: str

# parent node

def generate_answer(state: State):
    response = llm.invoke(state["question"])

    return {
        "answer_eng": response.content,
        "input_text": response.content,   # Pass to subgraph
    }


# subgraph node

def translate(state: State):
    prompt = f"""
Translate the following text into natural Nepali.

Text:
{state["input_text"]}
"""

    response = llm.invoke(prompt)

    return {
        "translated_text": response.content
    }


graph = StateGraph(State)

graph.add_node("generate_answer", generate_answer)

graph.add_node("translation", translate)

graph.add_edge(START, "generate_answer")
graph.add_edge("generate_answer", "translation")
graph.add_edge("translation", END)

chat = graph.compile()


initial_state = {
    "question": "Tell me about Mount Everest."
}

result = chat.invoke(initial_state)

print("\nQuestion:")
print(result["question"])

print("\nEnglish Answer:")
print(result["answer_eng"])

print("\nNepali Translation:")
print(result["translated_text"])