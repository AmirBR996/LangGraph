from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You have great knowledge about history.
Give a short answer to the following question:
{question}
"""
)

model = ChatGroq(model_name="llama-3.3-70b-versatile")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    question = state["messages"][-1].content

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer the question",
        "question": question,
        "instruction": "Approve this question? yes/no"
    })

    if str(decision).strip().lower() == "no":
        return {
            "messages": [
                AIMessage(content="Question was rejected by the reviewer.")
            ]
        }

    formatted_prompt = prompt.format(question=question)
    response = model.invoke(formatted_prompt)
    return {"messages": [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}

question = "Who was the first prime minister of Nepal?"

chatbot.invoke(
    {"messages": [HumanMessage(content=question)]},
    config=config
)

user_input = input(f"Do you want to approve this question — '{question}'? yes/no: ")

result = chatbot.invoke(
    Command(resume=user_input),
    config=config
)

print(result["messages"][-1].content)