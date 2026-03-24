from typing import Optional, cast, Any

import logging
from .valkeyConnect import create_valkey_client
from .connectQdrant import create_qdrant_client
from .pgSync import get_conn_str
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from glide import GlideClient
from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

class DBRegistry:
	def __init__(self):
		self.valkey_client: Optional[GlideClient] = None
		self.qdrant_client: Optional[AsyncQdrantClient] = None
		self.pg_pool: Optional[AsyncConnectionPool] = None
		self.checkpointer: Optional[AsyncPostgresSaver] = None

	async def startup(self):
		"""Initialize all the database clients"""
		logger.info("Initializing all DB clients")
		
		# Initialize Valkey
		try:
			self.valkey_client = await create_valkey_client()
			logger.info("Valkey client initialized")
		except Exception as e:
			logger.error(f"Failed to initialize Valkey: {e}")
			# raise

		# Initialize Qdrant
		try:
			self.qdrant_client = create_qdrant_client()
			# Verify connection? Qdrant client is lazy or init is fast.
			logger.info("Qdrant client initialized")
		except Exception as e:
			logger.error(f"Failed to initialize Qdrant: {e}")
			# raise

		# Initialize Postgres and Checkpointer
		try:
			# open=False is default in constructor if using 'with', but here we manage updated lifecycle
			# Using open=False creates the pool but doesn't open connections until 'open()' or context manager.
			# However, if we don't use context manager, we must call open() or use it directly?
			# AsyncConnectionPool(..., open=True) to start immediately?
			# Docs say: "If open is True, the pool is opened on creation. If False, it should be opened calling open()."
			self.pg_pool = AsyncConnectionPool(
				conninfo=get_conn_str(), 
				max_size=10, 
				open=False, # We will open it explicitly
				kwargs={"row_factory": dict_row}
			)
			await self.pg_pool.open()
			
			typed_pool = cast(AsyncConnectionPool[AsyncConnection[dict[str, Any]]], self.pg_pool)
			self.checkpointer = AsyncPostgresSaver(typed_pool)
			await self.checkpointer.setup()
			logger.info("Postgres Checkpointer initialized")

		except Exception as e:
			logger.error(f"Failed to initialize Postgres: {e}")
			# Clean up others if PG fails?
			# raise

	async def shutdown(self):
		"""Close all connections gracefully"""
		logger.info("Shutting down DB clients")

		if self.valkey_client:
			await self.valkey_client.close()
			logger.info("Valkey client closed")

		if self.qdrant_client:
			# AsyncQdrantClient has a close method
			await self.qdrant_client.close()
			logger.info("Qdrant client closed")

		if self.pg_pool:
			await self.pg_pool.close()
			logger.info("Postgres pool closed")