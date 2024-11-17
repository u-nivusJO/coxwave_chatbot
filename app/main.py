from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.rag import RAGSystem
from contextlib import asynccontextmanager
import json
import os
import uvicorn

rag_system = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_system.initialize()
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        raise
    yield

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=os.path.join("app", "templates"))

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def generate_response():
        try:
            response = rag_system.query(request.question)
            yield f"data: {json.dumps({'content': response})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )
 
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)