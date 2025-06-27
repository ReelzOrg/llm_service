from typing import Any
import aiohttp
import asyncio

from enum import StrEnum

session = None

class Mode(StrEnum):
  GET = "get"
  POST = "post"
  PUT = "put"
  DELETE = "delete"

async def initSession():
  global session
  if session is None:
    session = aiohttp.ClientSession()

async def close_session():
    global session
    if session is not None:
        await session.close()
        session = None

## sorry I am more comfortable with the fetch function in javascript and had to make it this way
async def fetch(url: str, mode: Mode = Mode.GET, headers: dict = None, jsonPayload: dict = None) -> str:
  await initSession()
  if mode == Mode.GET:
    async with session.get(url, headers=headers) as response:
      return await response.text()
  elif mode == Mode.POST:
    async with session.post(url, headers=headers, json=jsonPayload) as response:
      if response.status == 200:
        data = await response.json()
        return data.get("response", "")
      else:
        return await response.text()
  elif mode == Mode.PUT:
    async with session.put(url, headers=headers, json=jsonPayload) as response:
      return await response.text()
  elif mode == Mode.DELETE:
    async with session.delete(url, headers=headers) as response:
      return await response.text()

async def fetch_stream(url: str, headers: dict = None, jsonPayload: dict = None):
  await initSession()
  async with session.post(url, headers=headers, json=jsonPayload) as response:
    # iter_any is used here instead of iter_chunked because iter_any sends the data faster
    async for chunk in response.content.iter_any():
      if chunk:
        yield chunk

async def send_heartbeats(url: str, session_id: str, jsonPayload: Any):
  """Send periodic requests to keep ollama model loaded"""
  while session_id in active_sessions:
    try:
      await fetch_stream(
        url,
        jsonPayload=jsonPayload
        # jsonPayload={"model": model_name, "messages": [], "stream": False}
      )
    except Exception:
      pass
    await asyncio.sleep(300)  #Every 5 minutes