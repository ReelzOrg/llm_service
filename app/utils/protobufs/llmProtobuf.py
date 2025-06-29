import os
import grpc
import time
import logging
import uuid
import json
from typing import Dict, List
from concurrent import futures
import grpc.aio
import asyncio

from protos import llm_service_pb2, llm_service_pb2_grpc
from utils.networkReq import fetch_stream, initSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:4b")

MAX_HISTORY = 20
# Use a database for proper storage of the chats instead of a simple dictionary
chat_history: Dict[str, List[Dict[str, str]]] = {}

class LLMServiceClass(llm_service_pb2_grpc.LLMServiceServicer):
  """ Implements the gRPC LLMService service methods """

  def __init__(self) -> None:
    super().__init__()
    logging.info("LLMServiceClass initialized.")

  # Corresponds to `rpc LLMChat(LLMRequest) returns (stream LLMResponse) {}` in .proto
  async def LLMChat(self, request, context):
    prompt, session_id, history = request.prompt, request.session_id, request.history
    model = request.model if request.model else DEFAULT_OLLAMA_MODEL
    if not session_id:
      session_id = str(uuid.uuid4())
      # active_sessions.add(session_id)

    curr_chat_messages = chat_history.get(session_id, [])

    # We are not sending the history from the client side for now
    # If history was provided in the request, use that (or merge)
    # change this code to add the history to the database and use a proper merging logic
    if history:
      # we are using the history as the only truth if the client sends it but it will be more secure if maybe the client
      # sends a token or somthing which can fetch the history from the database
      curr_chat_messages = []
      for msg in history:
        curr_chat_messages.append({"role": msg.role, "content": msg.content})

    curr_chat_messages.append({"role": "user", "content": prompt})
    curr_chat_messages = curr_chat_messages[-MAX_HISTORY:]

    ollama_payload = {"model": model, "messages": curr_chat_messages, "stream": True}

    full_response_content = ""
    buffer = "" # Buffer for incomplete JSON chunks

    #context: A grpc.ServicerContext object, providing information about the RPC (e.g., deadlines, metadata, cancellation).
    try:
      async for chunk in fetch_stream(OLLAMA_CHAT_ENDPOINT, jsonPayload=ollama_payload):
        chunk_str = chunk.decode("utf-8")
        buffer += chunk_st

        while '\n' in buffer:
          line, buffer = buffer.split('\n', 1)
          if line.strip():
            try:
              data = json.loads(line)
              if 'message' in data and 'content' in data['message']:
                content_part = data['message']['content']
                full_response_content += content_part
                yield llm_service_pb2.LLMResponse(content=content_part)
              elif 'done' in data and data['done']:
                pass
            except json.JSONDecodeError:
              print(f"Warning: Could not decode JSON line: {line}")
              yield llm_service_pb2.LLMResponse(content=f"error: [Error: Failed to process Ollama response chunk]")

      # If there's anything left in the buffer after the stream ends, it's an incomplete line.
      # Probably write a function that adds the response to the full_response_content variable as the code is same
      if buffer.strip():
        try:
          data = json.loads(buffer)
          if 'message' in data and 'content' in data['message']:
            content_part = data['message']['content']
            full_response_content += content_part
            yield llm_service_pb2.LLMResponse(content=content_part)
        except json.JSONDecodeError:
          logging.warning(f"[{session_id}] Warning: Incomplete JSON buffer at end: {buffer}")
          yield llm_service_pb2.LLMResponse(content="[Error: Incomplete response from Ollama]\n")

    except Exception as e:
      error_message = f"An unexpected server error occurred: {e}"
      print(error_message)
      yield llm_service_pb2.LLMResponse(content=f"error: [Error: {error_message}]")

    if full_response_content:
      curr_chat_messages.append({"role": "assistant", "content": full_response_content})
      chat_history[session_id] = curr_chat_messages
      logging.info(f"[{session_id}] Full response added to history.")
    else:
      logging.warning(f"[{session_id}] No content received from Ollama for history persistence.")

async def serve():
  await initSession()
  server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
  llm_service_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceClass(), server)

  # Tells the server to listen on the specified port without requiring SSL/TLS encryption.
  # For production, you'd use server.add_secure_port with credentials.
  server.add_insecure_port("[::]:50051")
  await server.start()
  await server.wait_for_termination()
  logging.info("LLMService started on port 50051")

if __name__ == '__main__':
  asyncio.run(serve())

"""
Important: In gRPC streaming, the client will only receive the streaming responses. If you generate a new session ID
on the server, you need a mechanism to send that back to the client. For this streaming RPC, it's not straightforward
to send an initial metadata message with the new ID. A common pattern is:
  1. For the first request in a session, the client sends an empty session_id.
  2. The server generates it.
  3. The server's first yielded message could contain a special field or a specific format to indicate the new
session ID. Or, the client could always send a generated ID from its side. For simplicity, I've left it such that
if the client doesn't send it, the server generates and uses it internally, but the client won't know it unless you
explicitly encode it in the streamed content or use gRPC metadata (which is more complex for the first yield).
"""