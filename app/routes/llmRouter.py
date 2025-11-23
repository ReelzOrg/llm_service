import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.utils.networkReq import fetch, fetch_stream

## /llm
router = APIRouter()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "qwen3:4b")

MAX_HISTORY = 20
active_sessions = set()

# Use a database for proper storage of the chats instead of a simple dictionary
chat_history: Dict[str, List[Dict[str, str]]] = {}

async def getPromptFromClient(req: Request):
  req_body = await req.json()
  messages = req_body.get("messages") #{"messages": [{"role": "user", "content": "Tell me a joke"}]}
  model = req_body.get("model", DEFAULT_OLLAMA_MODEL)

  if not messages or not isinstance(messages, list):
    raise HTTPException(status_code=400, detail="'messages' field is required and must be a list.")

  # Convert messages array to a string prompt for Ollama
  prompt = "\n".join([f"{msg['content']}" for msg in messages])
  ollama_payload = {"model": model,"prompt": prompt,"stream": True}
  return ollama_payload

class ChatModel(BaseModel):
  prompt: str
  model: Optional[str] = DEFAULT_OLLAMA_MODEL
  session_id: Optional[str] = None  # optional unique ID to maintain conversation history.

## Endpoint for single chat request
@router.get("/ask")
def modelInterface(request: Request):
  templates = Jinja2Templates(directory="templates")
  return templates.TemplateResponse("ask.html", { "request": request, "message": "Hello" })

@router.post("/ask")
async def askModel(req: Request):
  """To ask the model basic, one-off questions without context"""

  # get the logged in user id from the jwt token that will be passed from the frontend
  try:
    ollama_payload = await getPromptFromClient(req)
    return StreamingResponse(
      fetch_stream(OLLAMA_GENERATE_ENDPOINT, jsonPayload=ollama_payload),
      media_type="text/event-stream"
    )
  except:
    return { "success": False, "message": "Something went wrong" }

## Endpoint for continued converstion (context is included)
@router.get("/chat")
async def chatInterface(request: Request):
  templates = Jinja2Templates(directory="templates")
  return templates.TemplateResponse("ask.html", { "request": request, "message": "Yo" })

# This is now handled by the gRPC function in llmProtobuf.py
# When a user waits for a long time before sending another prompt, ollama will unload the model from memory to save
# resources. To prevent this, we will keep the model loaded in memory by sending a keep-alive request every 5 minutes.
# Or we can configure ollama before running - `ollama serve --keep-alive 30m`
@router.post("/chat")
async def chatWithModel(req: ChatModel):
  """When the user wants to send multiple queries and maintaining context is needed"""

  # Generate a new session ID if one is not provided in the request
  # session_id = req.session_id if req.session_id else str(uuid.uuid4())
  if req.session_id:
    session_id = req.session_id
    #session_id was sent by the client hence we dont have to add to active_sessions since we have already done this
    #although, check if the session_id is a valid uuid4, if no then maybe the user tampered with the session id.
    #in this case create a new one
  else:
    session_id = str(uuid.uuid4())
    active_sessions.add(session_id)

  model_name = req.model
  prompt = req.prompt

  print("Prompt received:", prompt)

  curr_chat_messages = chat_history.get(session_id, [])
  curr_chat_messages.append({"role": "user", "content": prompt})
  curr_chat_messages = curr_chat_messages[-MAX_HISTORY:]

  ollama_payload = {
    "model": model_name,
    "messages": curr_chat_messages,
    "stream": True
  }

  # add a buffer here to store incomplete chunks. During any streaming, the chunks might be incomplete which will lead
  # to incomplete json and hence an error. So add a buffer which will keep appending new chunks until the complete
  # line is received
  async def generate_ollama_stream():
    full_response_content = ""

    try:
      async for chunk in fetch_stream(OLLAMA_CHAT_ENDPOINT, jsonPayload=ollama_payload):
        lines = chunk.decode("utf-8").splitlines()
        for line in lines:
          if line.strip():
            try:
              data = json.loads(line)
              if 'message' in data and 'content' in data['message']:
                content_part = data['message']['content']
                full_response_content += content_part
                yield f"{content_part}\n\n"
              elif 'done' in data and data['done']:
                pass
            except json.JSONDecodeError:
              print(f"Warning: Could not decode JSON line: {line}")
              yield f"error: [Error: Failed to process Ollama response chunk]\n\n"
    except Exception as e:
      error_message = f"An unexpected server error occurred: {e}"
      print(error_message)
      yield f"error: [Error: {error_message}]\n\n"

    if full_response_content:
      curr_chat_messages.append({"role": "assistant", "content": full_response_content})
      chat_history[session_id] = curr_chat_messages # Persist updated history

  return StreamingResponse(
    generate_ollama_stream(),
    media_type="text/event-stream",
    headers={"X-Session-ID": session_id}
  )
