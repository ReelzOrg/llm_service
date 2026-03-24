import os
from dotenv import load_dotenv
if not load_dotenv():
	raise Exception("Could not load env")

import asyncio
from typing import Optional
from glide import BackoffStrategy, NodeAddress, GlideClient, GlideClientConfiguration

async def create_valkey_client() -> GlideClient:
	"""Creates and returns a new Valkey client instance"""
	host = os.getenv("VALKEY_HOST", "localhost")
	port = int(os.getenv("VALKEY_PORT", 6379))
	
	addresses = [NodeAddress(host, port)]
	config = GlideClientConfiguration(
		addresses=addresses,
		request_timeout=500,
		client_name="AI_cache",
		# reconnect_strategy=BackoffStrategy(num_of_retries=5, factor=100)
	)
	return await GlideClient.create(config)
