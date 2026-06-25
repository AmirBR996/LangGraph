from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests

load_dotenv()

model = ChatGroq(model_name="llama-3.1-8b-instant")


@tool
def get_weather(city: str) -> str:
    """Returns current weather for a given city using wttr.in API."""
    response = requests.get(f"https://wttr.in/{city}?format=3")
    if response.status_code == 200:
        return response.text
    return f"Could not fetch weather for {city}."


tools = [get_weather]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: ChatState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def hitl_node(state: ChatState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    decision = interrupt({
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "instruction": "Approve this tool call? yes/no"
    })

    if str(decision).strip().lower() == "no":
        return {
            "messages": [
                ToolMessage(
                    content="Tool call was rejected by the reviewer.",
                    tool_call_id=tool_call["id"]
                )
            ]
        }

    result = get_weather.invoke(tool_call["args"])
    return {
        "messages": [
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        ]
    }


def should_use_tool(state: ChatState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "hitl_node"
    return "human_node"


def human_node(state: ChatState):
    user_input = interrupt("Your message (or type 'exit' to quit): ")
    if user_input.strip().lower() == "exit":
        return {"messages": [AIMessage(content="Goodbye!")]}
    return {"messages": [HumanMessage(content=user_input)]}


def should_continue(state: ChatState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.content == "Goodbye!":
        return END
    return "agent_node"


checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("agent_node", agent_node)
graph.add_node("hitl_node", hitl_node)
graph.add_node("human_node", human_node)
graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", should_use_tool)
graph.add_edge("hitl_node", "agent_node")
graph.add_conditional_edges("human_node", should_continue)

chatbot = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}

first_input = input("You: ")

result = chatbot.invoke(
    {"messages": [HumanMessage(content=first_input)]},
    config=config
)

while True:
    last = result["messages"][-1]

    if isinstance(last, AIMessage) and last.content == "Goodbye!":
        print("Assistant: Goodbye!")
        break

    if hasattr(last, "tool_calls") and last.tool_calls:
        tool_call = last.tool_calls[0]
        print(f"\n[Tool Request] {tool_call['name']} with args {tool_call['args']}")
        approval = input("Approve this tool call? yes/no: ")
        result = chatbot.invoke(Command(resume=approval), config=config)

    elif isinstance(last, AIMessage):
        print(f"Assistant: {last.content}")
        user_input = input("You: ")
        result = chatbot.invoke(Command(resume=user_input), config=config)