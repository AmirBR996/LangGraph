from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

load_dotenv()

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",  
    temperature=0
)

# Search Tool
search = DuckDuckGoSearchRun(region="us-en")

@tool
def search_tool(query: str) -> str:
    """
    Search the web for latest information using DuckDuckGo.
    Useful for finding current events, news, and recent information.
    Always use this when asked about recent events or current information.
    """
    try:
        result = search.run(query)
        return result if result else "No search results found"
    except Exception as e:
        return f"Search failed: {str(e)}"

# Calculator Tool
@tool
def calculator(firstnum: float, secondnum: float, operation: str) -> dict:
    """
    Perform basic math operations.
    """

    if operation == "add":
        result = firstnum + secondnum

    elif operation == "sub":
        result = firstnum - secondnum

    elif operation == "multiply":
        result = firstnum * secondnum

    elif operation == "divide":
        if secondnum == 0:
            return {"error": "Cannot divide by zero"}

        result = firstnum / secondnum

    else:
        return {"error": "Invalid operation"}

    return {
        "first_number": firstnum,
        "second_number": secondnum,
        "operation": operation,
        "result": result
    }

# Tools list
tools = [search_tool, calculator]

# Bind tools with LLM
llm_with_tools = llm.bind_tools(tools)

# State
class ToolState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat node
def chat_model(state: ToolState):
    messages = state["messages"]
    
    # Add system message to guide tool usage
    system_message = SystemMessage(
        content="""You are a helpful assistant with access to tools.
When the user asks about recent events or current information, use the search_tool to find the latest information.
For math operations as add , sub , multiply , divide, use the calculator tool.
Always try to use the available tools to provide accurate and current information."""
    )
    
    full_messages = [system_message] + messages

    response = llm_with_tools.invoke(full_messages)

    return {
        "messages": [response]
    }

# Tool node
tool_node = ToolNode(tools)

# Graph
graph = StateGraph(ToolState)

graph.add_node("chatnode", chat_model)
graph.add_node("tool_node", tool_node)

# Start
graph.add_edge(START, "chatnode")

# Conditional routing
graph.add_conditional_edges(
    "chatnode",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)

# Return back after tool execution
graph.add_edge("tool_node", "chatnode")

# Compile
chatbot = graph.compile()

# Invoke
response = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content="who is the present pm of nepal"
            )
        ]
    }
)

# Print final response
print(response["messages"][-1].content)