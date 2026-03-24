import aiohttp
import asyncio

class HTTPClient:
  def __init__(self):
    self.session: aiohttp.ClientSession | None = None

  async def init(self):
    if self.session is None:
      # Increase pool limits for massive concurrency
      conn = aiohttp.TCPConnector(limit=1000, ssl=False)
      self.session = aiohttp.ClientSession(connector=conn)

  async def close(self):
    if self.session:
      await self.session.close()

  async def get(self, url, headers=None, **kwargs):
    print("GETTING URL: ", url)
    return await self._request("GET", url, headers=headers, **kwargs)

  async def post(self, url, data=None, **kwargs):
    return await self._request("POST", url, data=data, **kwargs)

  async def getMultiple(self, urls, headers=None, **kwargs):
    tasks = [self.get(url, headers=headers, **kwargs) for url in urls]
    return await asyncio.gather(*tasks)

  async def _request(self, method, url, data=None, headers=None, **kwargs):
    if not self.session:
      # raise RuntimeError("HTTPClient not initialized")
      print("Initializing HTTPClient")
      await self.init()

    final_headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      **(headers or {}),
    }
    
    if self.session:
      async with self.session.request(method, url, headers=final_headers, data=data, **kwargs) as resp:
        resp.raise_for_status()
        return await resp.json()
    else:
      # This is redundant since we are checking at the top of the function
      raise RuntimeError("HTTPClient not initialized")

http_client = HTTPClient()
# http_client.init()