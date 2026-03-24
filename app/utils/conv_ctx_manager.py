from datetime import datetime
import logging
import json
from typing import Any, Optional

from glide import ExpirySet, ExpiryType, GlideClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from qdrant_client import AsyncQdrantClient

from app.graph.state import RetrievedChunk

logger = logging.getLogger(__name__)

class ConversationContextManager:
	"""
	Manages conversation context and RAG retrieval.

	Responsibilities:
	1. Load conversation history from Postgres/Valkey
	2. Retrieve relevant chunks from Qdrant
	3. Assemble final context for LLM
	4. Manage token limits
	"""

	def __init__(self, qdrant_client: AsyncQdrantClient, valkey_client: GlideClient, collection_name: str = "user_documents"):
		self.qdrant_client = qdrant_client
		self.valkey_client = valkey_client
		self.collection_name = collection_name

		logger.info("ContextManager initialized")

	async def load_history(self, user_id: str, session_id: str, max_tokens: int = 4000) -> list[BaseMessage]:
		"""
		Load conversation history from cache/DB.

		Strategy:
		1. Check Valkey cache first (fast)
		2. Fall back to Postgres if not cached
		3. Trim to fit within max_tokens
		"""

		cache_key = f"chat_history:{user_id}:{session_id}"
        
		# Try Redis cache first
		cached = await self.valkey_client.get(cache_key)
		if cached:
			messages_data = json.loads(cached)
			messages = self._deserialize_messages(messages_data)
			logger.info(f"Loaded {len(messages)} messages from cache")
			return self._trim_messages(messages, max_tokens)

		# TODO: Load from Postgres if not cached
		# For now, return empty
		logger.info("No cached history found, returning empty")
		return []

	async def save_history(self, user_id: str, session_id: str, messages: list[BaseMessage]):
		"""Save conversation history to cache."""

		cache_key = f"chat_history:{user_id}:{session_id}"

		# Serialize messages
		messages_data = self._serialize_messages(messages)

		# Store with 1 hour TTL
		await self.valkey_client.set(cache_key,json.dumps(messages_data), expiry=ExpirySet(ExpiryType.SEC, 3600))

		logger.info(f"Saved {len(messages)} messages to cache")

	async def retrieve_context(self, query: str, user_id: str, max_chunks: int = 5, similarity_threshold: float = 0.7) -> list[RetrievedChunk]:
		"""
		Retrieve relevant chunks from vector DB.
		Uses:
		- User's uploaded documents
		- Previous conversation context
		- Shared knowledge base (if applicable)
		""" 
		try:
			# TODO: Generate embedding for query
			# For now, return empty
			# query_embedding = await self._embed_text(query)
			# Search Qdrant
			# results = await self.qdrant.search(
			#     collection_name=self.collection_name,
			#     query_vector=query_embedding,
			#     query_filter=Filter(
			#         must=[
			#             FieldCondition(
			#                 key="user_id",
			#                 match=MatchValue(value=user_id)
			#             )
			#         ]
			#     ),
			#     limit=max_chunks,
			#     score_threshold=similarity_threshold
			# )
			# chunks = []
			# for result in results:
			#     chunks.append({
			#         "chunk_id": result.id,
			#         "content": result.payload["content"],
			#         "source_file_id": result.payload["file_id"],
			#         "relevance_score": result.score,
			#         "chunk_type": result.payload["type"],
			#         "metadata": result.payload.get("metadata", {})
			#     })
			# logger.info(f"Retrieved {len(chunks)} chunks for query")
			# return chunks

			return []

		except Exception as e:
			logger.error(f"Error retrieving context: {e}")
			return []

	def _serialize_messages(self, messages: list[BaseMessage]) -> list[dict]:
		"""Convert LangChain messages to JSON-serializable format."""
		return [
		{
			"type": msg.__class__.__name__,
			"content": msg.content,
			"additional_kwargs": msg.additional_kwargs
		}
		for msg in messages
	]

	def _deserialize_messages(self, data: list[dict]) -> list[BaseMessage]:
		"""Convert JSON back to LangChain messages."""
		messages = []
		for item in data:
			if item["type"] == "HumanMessage":
				messages.append(HumanMessage(content=item["content"]))
			elif item["type"] == "AIMessage":
				messages.append(AIMessage(content=item["content"]))
			elif item["type"] == "SystemMessage":
				messages.append(SystemMessage(content=item["content"]))
		return messages

	def _trim_messages(self, messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
		"""
		Trim messages to fit within token limit.

		Strategy:
		- Keep system message
		- Keep most recent messages
		- Drop oldest messages first
		"""
	
		if not messages:
			return []
	
		# Rough token estimation (4 chars per token)
		estimated_tokens = sum(len(msg.content) // 4 for msg in messages)
	
		if estimated_tokens <= max_tokens:
			return messages
	
		# Keep system message if present
		system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
		other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
	
		# Trim from oldest
		trimmed: list[BaseMessage] = []
		current_tokens = sum(len(msg.content) // 4 for msg in system_msgs)
	
		for msg in reversed(other_msgs):
			msg_tokens = len(msg.content) // 4
			if current_tokens + msg_tokens <= max_tokens:
				trimmed.insert(0, msg)
				current_tokens += msg_tokens
			else:
				break

		logger.info(f"Trimmed messages from {len(messages)} to {len(trimmed) + len(system_msgs)}")

		return system_msgs + trimmed