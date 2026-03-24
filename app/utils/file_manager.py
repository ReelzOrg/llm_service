"""Tracks file processing status"""

from datetime import datetime
import logging
import json
from typing import Any, Optional

from glide import ExpirySet, ExpiryType, GlideClient

logger = logging.getLogger(__name__)

class FileManager:
	"""
	Manages file processing status.

	Uses Valkey for fast lookups.
	Ingestion workers update status in Valkey as they process files.
	"""

	def __init__(self, valkey_client: GlideClient):
		self.valkey_client = valkey_client
		logger.info(f"FileManager initialized")

	async def get_file_status(self, file_id: str) -> dict[str, Any]:
		"""
		Get processing status of a file. Returns a simple dictionary with 0
		progress if the file is not found in valkey and status as "pending".

		Returns:
		{
			"file_id": str,
			"status": "pending|processing|ready|failed",
			"progress": float,  # 0.0 to 1.0
			"message": str,
			"s3_uri": str,
			"filename": str,
			"mime_type": str,
			"size_bytes": int,
			"created_at": str,
			"completed_at": str | None
		}
		"""
		key = f"file_status:{file_id}"
		data = await self.valkey_client.get(key)

		if data:
			try:
				return json.loads(data)
			except json.JSONDecodeError:
				logger.error(f"Failed to decode JSON for key {key}")
				return {}
		else:
			return {
				"file_id": file_id,
				"status": "pending",
				"progress": 0.0,
				"message": "File not found in the cache",
				"s3_uri": "",
				"filename": "unknown",
				"mime_type": "",
				"size_bytes": 0,
				"created_at": datetime.now().isoformat(),
				"completed_at": None
			}

	async def update_file_status(self, file_id: str, status: str, progress: float = 0.0, message: str = "", **kwargs):
		"""
		Update file processing status
		Called by Ingestion Workers
		"""

		key = f"file_status:{file_id}"
		existing_data = await self.valkey_client.get(key)

		if existing_data:
			try:
				data = json.loads(existing_data)
			except:
				logger.error(f"Failed to decode JSON for key {key}")
				return {}
		else:
			data = {
				"file_id": file_id,
				"status": "pending",
				"progress": 0.0,
				"message": "",
				"s3_uri": "",
				"file_name": "unknown",
				"mime_type": "",
				"size_bytes": 0,
				"created_at": datetime.now().isoformat(),
				"completed_at": None
			}

		# Updating the fields
		data.update({
			"status": status,
			"progress": progress,
			"message": message,
			**kwargs
		})

		if status in ["ready", "failed"]:
			data["completed_at"] = datetime.now().isoformat()

		await self.valkey_client.set(key, json.dumps(data), expiry=ExpirySet(ExpiryType.SEC, 86400))
		logger.info(f"Updated file status: {file_id} -> {status} ({progress*100:.1f}%)")

	async def register_file(self, file_id: str, s3_uri: str, file_name: str, mime_type: str, size_bytes: str):
		"""Register a new file for processing"""
		await self.update_file_status(
			file_id=file_id,
			status="pending",
			progress=0.0,
			message="File registered, queued for processing",
			s3_uri=s3_uri,
			file_name=file_name,
			mime_type=mime_type,
			size_bytes=size_bytes
		)