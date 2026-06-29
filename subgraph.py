from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm

parent_llm = ChatGroq(model_name="llama-3.3-70b-versatile")
translator_llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# parent state

class ParentState(TypedDict):
    question: str
    answer_eng: str
    input_text: str
    translated_text: str


# subgraph state

class SubState(TypedDict):
    input_text: str
    translated_text: str


# parent node

def generate_answer(state: ParentState):
    response = parent_llm.invoke(state["question"])

    return {
        "answer_eng": response.content,
        "input_text": response.content,   # Pass to subgraph
    }


# subgraph node

def translate(state: SubState):
    prompt = f"""
Translate the following text into natural Nepali.

Text:
{state["input_text"]}
"""

    response = translator_llm.invoke(prompt)

    return {
        "translated_text": response.content
    }


# subgraph

translation_graph = StateGraph(SubState)

translation_graph.add_node("translate", translate)

translation_graph.add_edge(START, "translate")
translation_graph.add_edge("translate", END)

translation_graph = translation_graph.compile()

# parent graph

graph = StateGraph(ParentState)

graph.add_node("generate_answer", generate_answer)

graph.add_node("translation_subgraph", translation_graph)

graph.add_edge(START, "generate_answer")
graph.add_edge("generate_answer", "translation_subgraph")
graph.add_edge("translation_subgraph", END)

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