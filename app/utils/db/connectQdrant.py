import os
import logging
from dotenv import load_dotenv
if not load_dotenv():
  raise Exception("Failed to load .env file")

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff

logger = logging.getLogger(__name__)

def create_qdrant_client(**kwargs) -> AsyncQdrantClient:
	"""Creates and returns a new AsyncQdrantClient instance"""
	return AsyncQdrantClient(url=os.getenv("QDRANT_URL_GRPC"), prefer_grpc=True, host=os.getenv("HOST") or "localhost", **kwargs)

async def create_qdrant_collection(client: AsyncQdrantClient, collection_name: str, config: VectorParams):
	try:
		if await client.collection_exists(collection_name):
			raise Exception("Collection already exists")
			# return

		# we are using Qwen/Qwen3-Embedding-0.6B which creates vector of size 1024
		await client.create_collection(
			collection_name=collection_name,
			vectors_config=config or VectorParams(size=1024, distance=Distance.COSINE),
		)

	except Exception as e:
		logger.error(f"Error creating collection - {collection_name}: {e}")


# HuggingFaceEmbeddings uses SentenceTransformer under the hood
# embeddingModel = SentenceTransformer(
# 	"Qwen/Qwen3-Embedding-0.6B",
# 	model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
# 	tokenizer_kwargs={"padding_side": "left"}
# )
# embeddingModel = HuggingFaceEmbeddings(
# 	model_name="Qwen/Qwen3-Embedding-0.6B",
# 	model_kwargs={"device": torch.cuda.is_available() and "cuda" or "cpu", "trust_remote_code": True},
# 	encode_kwargs={"normalize_embeddings": True}
# )

# vectorStore = QdrantVectorStore(client=qdrantClient, collection_name="user_conversations", embedding=embeddingModel)