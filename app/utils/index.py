from langchain_core.messages import HumanMessage

from app.graph.state import ChatGraphState, IngestionGraphState
from protos import llm_service_pb2

def content_block_to_prompt(req: llm_service_pb2.LLMRequest):
	"""
	Converts the ContentBlock prompt type to HumanMessage for LLM
	"""
	user_text = ""
	inline_context: list[str] = []
	ingestion_jobs: list[IngestionGraphState] = []

	for block in req.prompt:
		kind = block.WhichOneof("data")
		
		if block.mime_type == "text/plain":
			user_text += block.text + "\n"

		# File ingestion
		else:
			ingestion_jobs.append({
				"user_id": req.user_id,
				"session_id": req.session_id,
				"mime_type": block.mime_type,
				"source_uri": block.uri,
				"size_bytes": -1,
				"status": "PROCESSING",
				"extracted_text": None,
				"chunks": None,
				"storage_decision": None,
				"s3_uri": None,
				"vector_ids": None,
				"postgres_id": None,
				"semantic_context": None
			})
		
	# ChatGraphState is not a list of objects because one user prompt will have 1 intent but can create many
	# knowledge chunks
	chatState: ChatGraphState = {
		"user_id": req.user_id,
		"session_id": req.session_id,
		"user_text": user_text.strip(),
		"inline_context": inline_context,
		"short_term_history": [],
		"retrieved_chunks": [],
		"selected_model": {"name": req.model, "parameters": {}, "capabilities": [], "tools": [], "max_context_tokens": None},
		"context_for_llm": [],
		"current_token_usage": 0,
		"answer": None,
		"artifact_intents": [],
	}

	return chatState, ingestion_jobs

import contextvars
import logging

_req_id_ctx = contextvars.ContextVar("request_id", default=None)