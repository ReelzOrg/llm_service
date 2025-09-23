import os
from langchain_core.tools import tool
import cohere

from utils.networkReq import session

_SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def rerank_results(query: str, results: list, top_n: int = 5, model="rerank-v3.5"):
  documents = [f"{r.get('title', '')}\n{r.get('description', '')}" for r in results]

  response = co.rerank(
    query=query,
    documents=documents,
    model=model,
    top_n=min(top_n, len(documents))
  )

  reranked_results = [results[res.index] for res in response.results]

  return reranked_results

@tool
async def searxng_search(query: str) -> str:
  """Searches the web using SearxNG to get recent information"""
  try:
    async with session.get(f"{_SEARXNG_URL}/search", params={"q": query, "format": "json"}) as response:
      response.raise_for_status()
      search_data = await response.json()
      results = search_data.get("results", [])

      reranked_results = rerank_results(query, results)

      results_text = ""
      for result in reranked_results:
        results_text += f"Title: {result.get('title', '')}\n"
        results_text += f"URL: {result.get('url', '')}\n"
        results_text += f"Description: {result.get('description', '')}\n\n"

      print("------------------SearxNG Search----------------------------")
      print(results_text)
      print("------------------Search Results END------------------------")

      return results_text
  except Exception as e:
    return str(e)