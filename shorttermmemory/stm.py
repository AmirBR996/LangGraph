import os
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# Define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

# Build graph
builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

# Create Postgres checkpointer
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)

# Use the checkpointer
with checkpointer:
    # Create database tables (safe to call every time)
    checkpointer.setup()

    chatbot = builder.compile(checkpointer=checkpointer)

    thread_id = "1"

    print("Type 'exit' to quit.\n")

    while True:
        message = input("You: ")

        if message.lower() in ["exit", "quit", "bye"]:
            break

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=message)
                ]
            },
            config=config,
        )

        print("AI:", result["messages"][-1].content)