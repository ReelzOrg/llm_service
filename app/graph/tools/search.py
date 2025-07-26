import os
from langchain_core.tools import tool

from utils.networkReq import session

_SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")

@tool
async def searxng_search(query: str) -> str:
  try:
    async with session.get(f"{_SEARXNG_URL}/search", params={"q": query, "format": "json"}) as response:
      response.raise_for_status()
      search_data = await response.json()

      results_text = ""
      for result in search_data.get("results", []):
        results_text += f"Title: {result.get('title', '')}\n"
        results_text += f"URL: {result.get('url', '')}\n"
        results_text += f"Description: {result.get('description', '')}\n\n"

      return results_text
  except Exception as e:
    return str(e)