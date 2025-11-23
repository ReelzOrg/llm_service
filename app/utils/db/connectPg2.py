import asyncpg
from typing import List, Optional, AsyncGenerator, Any

_pool: Optional[asyncpg.Pool] = None

async def init_pool(**kwargs):
  global _pool

  if _pool:
    print("Connection pool already initialized.")
    return

  try:
    _pool = await asyncpg.create_pool(**kwargs)
  except Exception as e:
    print(f"Failed to initialize connection pool: {e}")
    raise

async def close_pool():
  global _pool
	
  if _pool:
    await _pool.close()
    _pool = None
    print("Connection pool closed.")
  else:
    print("Connection pool is not initialized or already closed.")

async def get_pool() -> Optional[asyncpg.Pool]:
  if _pool is None:
    raise RuntimeError("Database pool is not initialized. Call `init_pool` first.")
  return _pool

async def fetch(sql: str, *params: Any) -> List[asyncpg.Record]:
	"""
	Executes a query and fetches all results.

	Args:
		sql (str): The raw SQL query with placeholders ($1, $2, etc.).
		*params: The parameters to substitute into the query.

	Returns:
		A list of asyncpg.Record objects.
	"""
	pool = await _get_pool()
	async with pool.acquire() as connection:
		return await connection.fetch(sql, *params)

async def fetch_one(sql: str, *params: Any) -> Optional[asyncpg.Record]:
	"""
	Executes a query and fetches the first result.

	Args:
		sql (str): The raw SQL query with placeholders ($1, $2, etc.).
		*params: The parameters to substitute into the query.

	Returns:
		A single asyncpg.Record object or None if no record is found.
    """
	pool = await _get_pool()
	async with pool.acquire() as connection:
		return await connection.fetchrow(sql, *params)

async def execute(sql: str, *params: Any) -> str:
	"""
	Executes a non-returning query (INSERT, UPDATE, DELETE).

	Args:
		sql (str): The raw SQL query with placeholders ($1, $2, etc.).
		*params: The parameters to substitute into the query.

	Returns:
		A string status of the command (e.g., "INSERT 0 1").
	"""
	pool = await _get_pool()
	async with pool.acquire() as connection:
		# Use a transaction for data-modifying operations to ensure atomicity
		async with connection.transaction():
			return await connection.execute(sql, *params)

async def stream_results(sql: str, *params: Any, prefetch_size: int = 100) -> AsyncGenerator[asyncpg.Record, None]:
	"""
	Executes a query and streams the results using a server-side cursor.

	This is memory-efficient for very large datasets as it doesn't load
	the entire result set into memory at once.

	Args:
		sql (str): The raw SQL query with placeholders ($1, $2, etc.).
		*params: The parameters to substitute into the query.
		prefetch_size (int): The number of rows to fetch from the database at a time.

	Yields:
		An asyncpg.Record object for each row in the result set.
	"""
	pool = await _get_pool()
	async with pool.acquire() as connection:
		# Transactions are required for using cursors
		async with connection.transaction():
			# The cursor will be automatically closed when the transaction ends
			cursor = connection.cursor(sql, *params, prefetch=prefetch_size)
			async for record in cursor:
				yield record