from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama-3.3-70b-versatile")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


conn = sqlite3.connect("chat.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "1"


def chat(message):
    config = {"configurable": {"thread_id": thread_id}}

    state = {"messages": [HumanMessage(content=message)]}

    for event in chatbot.stream(state, config=config):
        if "chat_node" in event:
            msg = event["chat_node"]["messages"][-1]
            print(msg.content, end="", flush=True)


if __name__ == "__main__":
    chat("what is my name")