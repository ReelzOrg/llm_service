from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.tools import tool
import cohere

# from utils.networkReq import session, initSession
from utils.networkUtils import http_client

_SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
coApiKey = os.getenv("CO_API_KEY")
co = cohere.ClientV2(coApiKey)

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
    url = f"{_SEARXNG_URL}/search?q={query}&format=json"
    url = url.replace(" ", "%20")
    response = await http_client.get(url)
    print("------------------SEARCH RESULTS------------------------\n", response)
    return response
  #   async with session.get(f"{_SEARXNG_URL}/search", params={"q": query, "format": "json"}) as response:
  #     response.raise_for_status()
  #     search_data = await response.json()
  #     results = search_data.get("results", [])
  #     print("------------------SEARCH RESULTS------------------------\n", results)

  #     reranked_results = rerank_results(query, results);
  #     # print("------------------RERANKED RESULTS------------------------\n", reranked_results)

  #     results_text = ""
  #     for result in reranked_results:
  #       results_text += f"Title: {result.get('title', '')}\n"
  #       results_text += f"URL: {result.get('url', '')}\n"
  #       results_text += f"Description: {result.get('description', '')}\n\n"

  #     print("\n------------------SearxNG Search----------------------------")
  #     print(results_text)
  #     print("------------------Search Results END------------------------")

  #     return results_text
  except Exception as e:
    print("There was an error in searxng_search tool -------------------:", e)
    return str(e)