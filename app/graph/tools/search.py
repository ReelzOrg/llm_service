import os
from langchain_core.tools import tool

from utils.networkReq import session

_SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")

@tool
async def searxng_search(query: str) -> str:
  """Searches the web using SearxNG to get recent information"""
  try:
    async with session.get(f"{_SEARXNG_URL}/search", params={"q": query, "format": "json"}) as response:
      response.raise_for_status()
      search_data = await response.json()

      results_text = ""
      for result in search_data.get("results", []):
        results_text += f"Title: {result.get('title', '')}\n"
        results_text += f"URL: {result.get('url', '')}\n"
        results_text += f"Description: {result.get('description', '')}\n\n"

      print("------------------SearxNG Search----------------------------")
      print(results_text)
      print("------------------Search Results END------------------------")

      return results_text
  except Exception as e:
    return str(e)