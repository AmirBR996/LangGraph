from chatbot import chatbot , HumanMessage
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


thread_id = "1"

@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=req.message)]}

    def generate():
        for chunk, metadata in chatbot.stream(
            state,
            config=config,
            stream_mode="messages"
        ):
            if hasattr(chunk, "content") and chunk.content:
                data = json.dumps({"chunk": chunk.content})
                yield f"data: {data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")