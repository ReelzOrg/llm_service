import os
import grpc
import logging
import uuid
from concurrent import futures
import grpc.aio
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from protos import llm_service_pb2, llm_service_pb2_grpc
from utils.networkReq import close_session, initSession
from graph.builder import graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"
# DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:4b")

class LLMServiceClass(llm_service_pb2_grpc.LLMServiceServicer):
  """ Implements the gRPC LLMService service methods """

  def __init__(self) -> None:
    super().__init__()
    logging.info("LLMServiceClass initialized with LangChain")

  async def LLMChat(self, request, context):
    session_id = request.session_id
    if not session_id:
      session_id = str(uuid.uuid4())
      logging.info(f"Generated new session_id: {session_id}")
      # active_sessions.add(session_id)

    # The `config` dictionary is how we pass session-specific information
    # to the LangGraph. The `thread_id` is used by the checkpointer to
    # load and save the correct conversation state.
    config = {"configurable": {"thread_id": session_id}}

    messages = []
    if hasattr(request, 'history') and request.history:
      messages = [HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content) for msg in request.history]
    messages.append(HumanMessage(content=request.prompt))

    # The input to the graph is a dictionary matching the GraphState schema.
    graph_input = {"messages": messages}

    try:
      async for event in graph.astream_events(
        graph_input,
        config=config,
        version="v2"
      ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
          chunk = event["data"]["chunk"]
          if chunk.content:
            full_assistant_response += chunk.content
            yield llm_service_pb2.LLMResponse(content=chunk.content)

          # Tool calls within the stream from LLM (Ollama's tool_calls are often in content stream)
          if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            logging.info(f"[{session_id}] LLM signaled tool call in stream: {chunk.tool_calls}")

        elif kind == "on_tool_start":
          tool_name = event["name"]
          tool_input = event["data"].get("input", {})
          logging.info(f"[{session_id}] LangChain calling tool: {tool_name} with input: {tool_input}")
          yield llm_service_pb2.LLMResponse(content=f"\n\nAI needs to perform a web search for '{tool_input.get('query', 'something')}'...\n")

        elif kind == "on_tool_end":
          tool_name = event["name"]
          tool_output_length = len(str(event["data"].get("output", ""))) # Convert to str for len
          logging.info(f"[{session_id}] Tool '{tool_name}' executed. Output length: {tool_output_length}.")
          # No need to yield tool output directly, LLM will process it for final answer.
          yield llm_service_pb2.LLMResponse(content=f"Search complete. Generating final answer...\n\n")

    except Exception as e:
      logging.error(f"[{session_id}] Error in LLMChat: {e}")
      yield llm_service_pb2.LLMResponse(content=f"Error: {str(e)}")

    logging.info(f"[{session_id}] LLMChat RPC complete.")

async def serve():
  await initSession()

  cpu_cores = os.cpu_count()
  max_workers = min(32, cpu_cores + 1 if cpu_cores else 1)
  server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=max_workers))
  llm_service_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceClass(), server)

  # Tells the server to listen on the specified port without requiring SSL/TLS encryption.
  # For production, you'd use server.add_secure_port with credentials.
  server.add_insecure_port("[::]:50051")
  await server.start()
  logging.info("LLMService started on port 50051")
  try:
    await server.wait_for_termination()
  except KeyboardInterrupt:
    await server.stop(0)
  finally:
    await close_session()

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
"""
LangChain's Internal Mechanism: RunnableWithMessageHistory intercepts this config dictionary. Before it actually
runs the self.base_chain, it internally calls the get_session_history function and
automatically extracts the session_id from the config dictionary to pass it as the first argument to
get_session_history.
"""