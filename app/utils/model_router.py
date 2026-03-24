import re
import logging
import numpy as np
from typing import Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.graph.state import QueryAnalysis

logger = logging.getLogger(__name__)

class ModelRouter:
	"""
	Routes Queries to the most appropriate model
	"""
	def __init__(self):
		self.transformer = SentenceTransformer("all-MiniLM-L6-v2")
		self.intent_examples = {
			"GENERAL_CONVERSATION": [
				"Hi", "Who are you?",
				"Tell me a joke",
				"What is the difference between a normal chair and an ergonomic chair",
				"Give me all the "
			],
			"CODING_ASSISTANCE": [
				"Create a React component for a login form",
				"Write a Python script to scrape a website",
				"What is micro-services architecture",
				"How do I solve this ERROR: Unable to resolve the package numpy"
			],
			"DATA_ANALYSIS": [
				"Summarize this report",
				"Extract the key dates from this text",
				"What is the sentiment of this review?",
				"Analyze the trend in these numbers"
			]
		}

		self.intent_embeddings = {}
		for intent, examples in self.intent_examples.items():
			self.intent_embeddings[intent] = self.transformer.encode(examples)

	def _detect_intent(self, query_embeddings, threshold = 0.4):
		"""
		Compares query embedding to all intent clusters. 
		Returns the closest intent.
		"""
		best_intent = "UNKNOWN"
		best_score = -1

		for intent, example_embeddings in self.intent_embeddings.items():
			similarities = cosine_similarity(query_embeddings.reshape(1, -1), example_embeddings)

			# max_score

	async def route(self, query_analysis: QueryAnalysis, user_preferences: Optional[Any]):
		pass