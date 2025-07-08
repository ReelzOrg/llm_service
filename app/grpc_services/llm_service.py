import os
from sys import version
from tabnanny import verbose
import grpc
import time
import logging
import uuid
import json
from typing import Dict, List
from concurrent import futures
import grpc.aio
import asyncio

# LangChain Imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from protos import llm_service_pb2, llm_service_pb2_grpc
from utils.networkReq import close_session, initSession
from tools.search import searxng_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:4b")

MAX_HISTORY = 20

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
if not SEARXNG_URL:
  logging.warning("SEARXNG_URL environment variable not set. SearXNG search functionality may be limited or fail.")

# In a production, this would be a persistent store (e.g., Dragonfly or a database)
store: Dict[str, InMemoryChatMessageHistory] = {}

class LLMServiceClass(llm_service_pb2_grpc.LLMServiceServicer):
  """ Implements the gRPC LLMService service methods """

  def __init__(self) -> None:
    super().__init__()
    logging.info("LLMServiceClass initialized with LangChain")

    self.llm = ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0.5)
    self.llm_with_tools = self.llm.bind_tools([searxng_search]) #ADD TOOLS
    self.prompt_template = ChatPromptTemplate.from_messages([
      ("system", "You are a conversational AI assistant. Your name is Emma. You have access to a web search tool (searxng_search). \
      Use it whenever current or factual information beyond your training data is required to answer the user's \
      question accurately. Prioritize using the tool for questions about recent events or specific current data. \
      If you use the tool, integrate the results naturally into your answer. If asked any coding related questiosn \
      You should always use the latest and recommended approach."),
      MessagesPlaceholder(variable_name="chat_history"), # For injecting memory
      ("user", "{input}")
    ])

    # the `|` operator in langchain is used to chain the operations. This is acustom implementation developed by langchain
    # here it send the output of the first variable to the next
    self.base_chain = self.prompt_template | self.llm_with_tools
    self.chain = RunnableWithMessageHistory(
      self.base_chain,
      self.getSessionHistory,
      input_messages_key="input",
      history_messages_key="chat_history",
      verbose=True  #comment this out later
    )

    self.tool_executor =  {
      # "web_search": web_search
    }

    # self.memory = ConversationBufferWindowMemory(
    #   memory_key="chat_history",
    #   return_messages=True,
    #   k=MAX_HISTORY
    # )
    # self.chain = RunnablePassthrough | self.prompt_template | self.llm_tools | StrOutputParser()

  def getSessionHistory(self, session_id, **kwargs):
    if session_id not in store:
      logging.info(f"[{session_id}] Creating new InMemoryChatMessageHistory for session.")
      store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

  async def LLMChat(self, request, context):
    session_id = request.session_id
    if not session_id:
      session_id = str(uuid.uuid4())
      logging.info(f"Generated new session_id: {session_id}")
      # active_sessions.add(session_id)

    user_prompt = request.prompt
    full_assistant_response = ""
    try:
      async for event in self.chain.astream_events(
        {"input": user_prompt},
        config={"configurable": {"session_id": session_id}},
        version="v1"
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

        elif kind == "on_chain_end":
          # This event indicates the entire chain (LLM, tool calls, final LLM call) has completed.
          # The final LLM response has been accumulated in full_assistant_response
          logging.info(f"[{session_id}] LangChain processing complete. Final AI response accumulated.")
          break # All chunks should be processed, break the loop

    except Exception as e:
      logging.error(f"[{session_id}] Error in LLMChat: {e}")
      yield llm_service_pb2.LLMResponse(content=f"Error: {str(e)}")
      full_assistant_response = ""

    logging.info(f"[{session_id}] LLMChat RPC complete.")

async def serve():
  await initSession()
  server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
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