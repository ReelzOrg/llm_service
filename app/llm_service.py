# To generate the proto files & the mypy (Type Stubs) files run the following command (from the llm_service directory):
# python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. --mypy_out=. --mypy_grpc_out=. protos/llm_service.proto
# uv run buf generate protos --path protos/llm/v1

# Try this prompt - find me 10 startups that currently have less than 10 people and they are looking for software engineer immediately

"""
If file < 50k tokens: Skip the vector DB. Just save text to Postgres/Redis and stuff it into the prompt.
If file is CSV/JSON: Don't embed. Load it into a Pandas dataframe or context. RAG doesn't work well with math related files. 
If file > 100k tokens: Return "Ready" immediately, start background indexing, but use a simple keyword search for the very first query if the index isn't ready yet.

1. Two LangGraphs, not one
Graph A — Chat Graph (sync)
Reads memory
Calls tools
Responds to user

Graph B — Ingestion Graph (async)
Fetches file
Extracts text
Chunks
Embeds
Writes to Qdrant
They communicate only via DB + object storage, never direct calls.

TODO:
 - We will not be using OCR becuase the scope doesn't need very accurate text recognition and VLMs do a good job anyway
 - Use DeepEval for LLM evaluations
 - Use GPTCache for caching
 - Arize Phoenix for observability
 - DSPy for prompt engineering
 - Guardrails-ai for adding guardrails

wsl --shutdown
diskpart
# Inside diskpart:
select vdisk file="C:\Path\To\Your\ext4.vhdx"
compact vdisk
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Awaitable, Optional, Union, cast
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from langchain_core.messages import BaseMessage
from langchain_core.runnables import history
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
import grpc
import logging
import uuid
from concurrent import futures
import grpc.aio
import asyncio
from langgraph.types import StateSnapshot
from qdrant_client import AsyncQdrantClient
from typing_extensions import AsyncIterator

from protos import llm_service_pb2, llm_service_pb2_grpc
from .graph.builder import build_graph
from .utils.db.DBRegistry import DBRegistry
from app.graph.state import ChatGraphState, FileReference, ModelRoutingDecision, QueryAnalysis, RetrievedChunk
from app.graph.tools.search import searxng_search
from app.utils.conv_ctx_manager import ConversationContextManager
from app.utils.file_manager import FileManager
from app.utils.model_router import ModelRouter
if not load_dotenv():
	raise Exception("Failed to load .env file")

# Add parent directory to path to allow importing protos when running from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMServiceClass(llm_service_pb2_grpc.LLMServiceServicer):
	"""
	Main gRPC servicer for LLM chat
	
	Responsibilities:
	- Parse incoming gRPC requests
	- Start the graph
	- Stream events and responses to client
	- Handle file processing status
	"""

	def __init__(
		self,
		chat_graph: CompiledStateGraph,
		model_router: ModelRouter,
		file_manager: FileManager,
		context_manager: ConversationContextManager,
		qdrant_client: AsyncQdrantClient
	) -> None:
		super().__init__()
		self.chat_graph = chat_graph
		self.model_router = model_router
		self.file_manager = file_manager
		self.context_manager = context_manager
		self.qdrant_client = qdrant_client

		logger.info("LLMServiceClass initialized")

	# 1. Calculate the number of tokens in the input prompt
	# 2. Check if the input prompt is within the model's context length
	# 3. If the input prompt is not within the model's context length, return the input prompt
	# 4. Checking for intent is important, if the user asks to summarize a 100k token document, sending it to embedding
	# model won't be helpful because only LLM can handle that

	# The gRPC sends stream of responses back to the client so we use AsyncIterator
	# If this return type is not mentioned than BasedPyright think this function uses return (which it doesn't)
	async def LLMChat(self, request: llm_service_pb2.LLMRequest, context: grpc.aio.ServicerContext) -> AsyncIterator[llm_service_pb2.LLMResponse]:
		""" Main gRPC method - streams the responses and events back to the client and starts the graph """

		request_id = request.request_id or str(uuid.uuid4())
		session_id = request.session_id or str(uuid.uuid4()) # we are already creating the session ID in Node.js so this step is not actually needed
		user_id = request.user_id
		# active_sessions.add(session_id)
		logger.info(f"[{request_id}] New chat request from user {user_id}")

		sequence = 0

		def next_sequence():
			nonlocal sequence
			sequence += 1
			return sequence

		# Emit StreamStart event to signal the start of the stream
		yield llm_service_pb2.LLMResponse(
			request_id=request_id,
			sequence=next_sequence(),
			start=llm_service_pb2.StreamStart(timestamp=datetime.now().isoformat())
		)

		# Parse user input
		user_text, file_refs = self._parse_request(request.prompt)
		logger.info(f"[{request_id}] User text: {user_text[:10]}...")
		logger.info(f"[{request_id}] Attached files: {len(file_refs)}")

		# Check file processing status
		files_ready = True
		for file_ref in file_refs:
			status = await self.file_manager.get_file_status(file_ref["file_id"])

			if status["status"] != "ready":
				files_ready = False

				# Send file status update
				yield llm_service_pb2.LLMResponse(
					request_id=request_id,
					sequence=next_sequence(),
					file_status=llm_service_pb2.FileProcessingStatus(
						file_id=file_ref["file_id"],
						filename=status.get("filename", "unknown"),
						status=self._map_file_status(status=status["status"]),
						progress_percent=status.get("progress", 0.0),
						message=status.get("message", "")
					)
				)

		# Analyze query to route models
		query_analysis = await self._analyze_query(user_text, file_refs)

		# Route to appropriate model
		routing_decision = self.model_router.route(query_analysis=query_analysis, user_preferences=request.model_prefs if request.HasField("model_prefs") else None)

		# Send Model selection event
		yield llm_service_pb2.LLMResponse(
			request_id=request_id,
			sequence=next_sequence(),
			model_selected=llm_service_pb2.ModelSelected(
				model_name=routing_decision["primary_model"]["name"],
				parameters={k: str(v) for k, v in routing_decision["primary_model"]["parameters"].items()},
				routing_reason=routing_decision["reasoning"],
			)
		)

		initial_state = await self._build_initial_state(
			request_id=request_id,
			session_id=session_id,
			user_id=user_id,
			user_text=user_text,
			file_refs=file_refs,
			routing_decision=routing_decision,
			context_options=request.context_options if request.HasField("context_options") else None
		)

		# TODO: Logic for selecting a model (the current router is slow since it uses LLM to select the model)
			# Use semantic routering to select the model
		# TODO: yield the model selection event

		# Building the graph state
		# graph_input: ChatGraphState = ChatGraphState(
		# 	user_id=request.user_id,
		# 	session_id=session_id,
		# 	user_text="".join(block.text for block in request.prompt if block.mime_type == "text/plain"),
		# 	current_token_usage=0,
		# 	inline_context=[],
		# 	short_term_history=[],
		# 	retrieved_chunks=[],
		# 	selected_model={"name": "", "parameters": {}, "capabilities": [], "tools": [searxng_search], "max_context_tokens": None},
		# 	context_for_llm=[],
		# 	artifact_intents=[],
		# 	answer=None,
		# )

		# The `thread_id` is used by the checkpointer to load and save the correct conversation state.
		# Do not add checkpoint_ns. It is managed by LangGraph nd is automatically set
		# Thread ID is like a hard drive and checkpoint_ns is like a partition/folder
		config: RunnableConfig = {"configurable": {"thread_id": session_id}}

		# messages = []
		# if hasattr(request, 'history') and request.history:
		#   messages = [HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content) for msg in request.history]
		# messages.append(HumanMessage(content=request.prompt))

		full_assistant_response = ""
		model_usage_map: dict[str, dict] = {}
		stream_reason = llm_service_pb2.COMPLETED

		try:
			# With astream, I can define my own events as compared to astrem_events which sends only the
			# LangChain internal events
			async for event in self.chat_graph.astream(initial_state, config=config):
				if "response_chunks" in event and event["response_chunks"]:
					for chunk in event["response_chunks"]:
						full_assistant_response += chunk
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							token=llm_service_pb2.TextToken(text=chunk)
						)

				if "response_text" in event and event["response_text"]:
					yield llm_service_pb2.LLMResponse(
						request_id=request_id,
						sequence=next_sequence(),
						message=llm_service_pb2.TextMessage(text=event["response_text"])
					)

				if "citations" in event and event["citations"]:
					for citation in event["citations"]:
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							citation=llm_service_pb2.Citation(
								citation_id=citation.get("citation_id", ""),
								source_uri=citation.get("source_uri", ""),
								excerpt=citation.get("excerpt", ""),
								relevance_score=citation.get("relevance_score", 0.0)
							)
						)

				if "pending_artifacts" in event and event["pending_artifacts"]:
					for artifact in event["pending_artifacts"]:
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							artifact_started=llm_service_pb2.ArtifactStarted(
								artifact_id=artifact.get("artifact_id", ""),
								type=self._map_artifact_type(artifact["type"]),
								mime_type=artifact.get("mime_type", "")
							)
						)

						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							artifact_completed=llm_service_pb2.ArtifactCompleted(
								artifact_id=artifact.get("artifact_id", ""),
								final_uri=artifact.get("final_uri", ""),
								metadata=artifact.get("metadata", {})
							)
						)

				if "tool_calls" in event and event["tool_calls"]:
					for tool_call in event["tool_calls"]:
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							tool_call=llm_service_pb2.ToolCall(
								tool_name=tool_call["tool_name"],
								call_id=tool_call["call_id"],
								input_json=tool_call["input_json"]
							)
						)

				if "tool_results" in event and event["tool_results"]:
					for tool_result in event["tool_results"]:
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							tool_result=llm_service_pb2.ToolResult(
								call_id=tool_result["call_id"],
								output_json=tool_result["output_json"],
								success=tool_result.get("success", True)
							)
						)

				if "errors" in event and event["errors"]:
					for error in event["errors"]:
						yield llm_service_pb2.LLMResponse(
							request_id=request_id,
							sequence=next_sequence(),
							error=llm_service_pb2.ErrorEvent(
								code="GRAPH_ERROR",
								message=error,
								recoverable=True,
								details={}
							)
						)

				if "warnings" in event and event["warnings"]:
					for warning in event["warnings"]:
						logger.warning(f"[{request_id}] Warning: {warning}")

			final_state: ChatGraphState = await self.chat_graph.aget_state(config)
			StateSnapshot
			# Emit the Execution summary after the response is completed
			yield llm_service_pb2.LLMResponse(
				request_id=request_id,
				sequence=next_sequence(),
				summary=self._create_execution_summary(final_state as ChatGraphState)
			)

			# Stream End event
			yield llm_service_pb2.LLMResponse(
				request_id=request_id,
				sequence=next_sequence(),
				end=llm_service_pb2.StreamEnd(
					# reason=llm_service_pb2.
					timestamp=str(int(time.time()))
				)
			)

		except asyncio.CancelledError:
			logger.warning(f"[{request_id}] Request cancelled by client")
			yield llm_service_pb2.LLMResponse(
				request_id=request_id,
				sequence=next_sequence(),
				end=llm_service_pb2.StreamEnd(
					reason=llm_service_pb2.CANCELLED,
					timestamp=str(int(time.time()))
				)
			)

		except Exception as e:
			logging.error(f"[{session_id}] Error in LLMChat: {e}")
			stream_reason = llm_service_pb2.ERROR
			yield llm_service_pb2.LLMResponse(
				request_id=request_id,
				sequence=next_sequence(),
				error=llm_service_pb2.ErrorEvent(
					code=e.__class__.__name__,
					message=str(e),
					recoverable=False
				)
			)
		
		finally:
			model_usages = []
			total_input = total_output = 0

			for model_name, usage in model_usage_map.items():
				total_input += usage["input"]
				total_output += usage["output"]

				model_usages.append(
					llm_service_pb2.ModelUsage(
						model_name=model_name,
						input_tokens=usage["input"],
						output_tokens=usage["output"],
						total_tokens=usage["input"] + usage["output"],
					)
				)
			
			if model_usages:
				yield llm_service_pb2.LLMResponse(
					request_id=request_id,
					sequence=next_sequence(),
					summary=llm_service_pb2.ExecutionSummary(
						model_usage=model_usages,
						total_input_tokens=total_input,
						total_output_tokens=total_output,
						total_tokens=total_input + total_output,
					)
				)

			# ---- STREAM END ----
			yield llm_service_pb2.LLMResponse(
				request_id=request_id,
				sequence=next_sequence(),
				end=llm_service_pb2.StreamEnd(
					timestamp=str(int(time.time())),
					reason=stream_reason
				)
			)

		logging.info(f"[{session_id}] LLMChat RPC complete.")

	async def CheckFileStatus(self, request: llm_service_pb2.FileStatusRequest, context: grpc.aio.ServicerContext) -> llm_service_pb2.FileStatusResponse:
		"""Check Processing Status of a File"""
		file_infos = []

		for file_id in request.file_ids:
			status = await self.file_manager.get_file_status(file_id)

			file_infos.append(
				llm_service_pb2.FileInfo(
					file_id=file_id,
					status=self._map_file_status(status["status"]),
					s3_uri=status.get("s3_uri", ""),
					size_bytes=status.get("size_bytes", 0),
					mime_type=status.get("mime_type", ""),
					processing_progress=status.get("progress", 0)
				)
			)

		return llm_service_pb2.FileStatusResponse(files=file_infos)

	async def GetChatHistory(self, request: llm_service_pb2.ChatHistoryRequest, context: grpc.aio.ServicerContext) -> llm_service_pb2.ChatHistoryResponse:
		"""Retrieve chat history for a session."""
		# # Query Postgres for history
		# query = """
		# 	SELECT message_id, role, content, timestamp, file_ids
		# 	FROM chat_messages
		# 	WHERE user_id = $1 AND session_id = $2
		# 	ORDER BY timestamp DESC
		# 	LIMIT $3
		# """
    
		# rows = await postgres.fetch(query, user_id, session_id, max_messages)
    
		# messages = []
		# for row in reversed(rows):  # Reverse to get chronological order
		# 	messages.append(
		# 		llm_service_pb2.ChatMessage(
		# 			message_id=row['message_id'],
		# 			role=row['role'],
		# 			content=row['content'],
		# 			timestamp=row['timestamp'].isoformat(),
		# 			file_ids=row['file_ids'] or []
		# 		)
		# )
    
		# return llm_service_pb2.ChatHistoryResponse(messages=messages)

	# The return type is int and not llm_service_pb2.FileStatus because it is not
	# a valid type at runtime, the protobuf enums are just integers in python
	def _map_file_status(self, status: str):
		"""Map internal status to proto enum."""

		mapping = {
			"pending": llm_service_pb2.QUEUED,
			"processing": llm_service_pb2.PROCESSING,
			"ready": llm_service_pb2.READY,
			"failed": llm_service_pb2.FAILED
		}

		return mapping.get(status, llm_service_pb2.FILE_STATUS_UNSPECIFIED)

	# Returns QueryAnalysis object
	def _analyze_query(self, user_text: str, file_refs: list[FileReference]):
		"""Analyze user query to determine routing"""
		pass

	def _parse_request(self, requestPrompt: RepeatedCompositeFieldContainer) -> tuple[str, list[FileReference]]:
		"""Extract text and file references from request."""

		user_text = ""
		file_refs: list[FileReference] = []

		for block in requestPrompt:
			if block.WhichOneof("data") == "text":
				user_text += block.text + "\n"
			
			elif block.WhichOneof("data") == "uri":
				file_name = self._extract_file_name(block.uri)
				file_refs.append(
					FileReference(
						file_id=file_name,
						s3_uri=block.uri,
						mime_type=block.mime_type,
						status="pending",
						processing_progress=0.0,
						thumbnail_uri=None
					)
				)

		return user_text.strip(), file_refs

	async def _build_initial_state(
		self,
		request_id: str,
		session_id: str,
		user_id: str,
		user_text: str,
		file_refs: list[FileReference],
		routing_decision: ModelRoutingDecision,
		context_options: Optional[llm_service_pb2.ContextOptions]
	) -> ChatGraphState:
		"""Build initial state for chat graph."""

		# Load Conversation history
		conversation_history: list[BaseMessage] = await self.context_manager.load_history(
			user_id=user_id,
			session_id=session_id,
			max_tokens=context_options.max_history_tokens if context_options else 4000
		)

		# Retrieve relavant context if needed
		retrieved_context: list[RetrievedChunk] = []
		if context_options and context_options.enable_retrieval:
			retrieved_context = await self.context_manager.retrieve_context(
				query=user_text,
				user_id=user_id,
				max_chunks=context_options.max_retrieval_chunks or 5
			)

		return ChatGraphState(
			user_id=user_id,
			session_id=session_id,
			request_id=request_id,
			user_text=user_text,
			attached_files=file_refs,
			conversation_history=conversation_history,
			retrieved_context=retrieved_context,
			routing_decision=routing_decision,
			selected_model=routing_decision["primary_model"],
			llm_messages=[],
			current_token_count=0,
			max_tokens_allowed=routing_decision["primary_model"]["capabilities"]["max_context_tokens"],
			files_ready=all(ref.status == "ready" for ref in file_refs),
			needs_retrieval=bool(retrieved_context),
			needs_vision_model=routing_decision.get("vision_model") is not None,
			needs_multi_step=False, #Dont hardcode this,
			response_text=None, #Or this
			response_chunks=[], #Or this
			citations=[],
			pending_artifacts=[],
			errors=[],
			warnings=[],
			timestamp=datetime.now().isoformat()
		)
	
	def _create_execution_summary(self, summary: ChatGraphState) -> llm_service_pb2.ExecutionSummary:
		# TODO: Extract actual metrics from state
		return llm_service_pb2.ExecutionSummary(
			model_usage=[],
			total_input_tokens=0,
			total_output_tokens=0,
			total_tokens=0,
			currency="USD",
			estimated_cost=0.0,
			files_processed=len(summary.get("attached_files", [])),
			retrieval_chunks_used=len(summary.get("retrieved_context", []))
		)

	def _extract_file_name(self, file_uri: str) -> str:
		"""example - https://reelzapp.s3.us-east-1.amazonaws.com/.../.../fileName.mp4 -> fileName"""
		return file_uri.split("/")[-1].split(".")[0]

async def serve():
	# DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/llmchats")
	
	registry = DBRegistry()
	await registry.startup()

	server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=os.cpu_count()))
	try:
		assert registry.valkey_client is not None
		assert registry.qdrant_client is not None
		assert registry.pg_pool is not None
		assert registry.checkpointer is not None

		# Initialize dependencies
		file_manager = FileManager(registry.valkey_client)
		model_router = ModelRouter()
		context_manager = ConversationContextManager(
			qdrant_client=registry.qdrant_client,
			valkey_client=registry.valkey_client
		)

		compiled_graph = build_graph(registry.checkpointer)

		llm_service = LLMServiceClass(
			chat_graph=compiled_graph,
			model_router=model_router,
			file_manager=file_manager,
			context_manager=context_manager,
			qdrant_client=registry.qdrant_client
		)

		# Injecting the compiled graph into the service
		llm_service_pb2_grpc.add_LLMServiceServicer_to_server(llm_service, server)

		# Tells the server to listen on the specified port without requiring SSL/TLS encryption.
		# For production, use server.add_secure_port with credentials.
		server.add_insecure_port("[::]:50051")
		await server.start()
		logging.info("LLMService started on port 50051")

		await server.wait_for_termination()
	except KeyboardInterrupt:
		# signal.signal(signal.SIGTERM, handle_sigterm)
		await server.stop(0)
	finally:
		await registry.shutdown()


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