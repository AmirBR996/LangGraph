from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model_name="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

def chat_node(state : ChatState):
    messages = state['messages']

    response = model.invoke(messages)

    return {'messages' : [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node" , chat_node)

graph.add_edge(START , 'chat_node')

graph.add_edge('chat_node' , END)

chatbot = graph.compile(checkpointer= checkpointer)

thread_id = "1"


def chat(message):
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=message)]}
    for chunk, metadata in chatbot.stream(
        state,
        config=config,
        stream_mode="messages"
    ):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    chat("give me essay on 500 words about dynamics")
