import time
from fastapi import Request

async def calcTimeTaken(req: Request, call_next):
  start_time = time.perf_counter()
  res = await call_next(req)
  process_time = time.perf_counter() - start_time
  res.headers["X-Process-Time"] = str(process_time)
  print(f"The time {req.url} took was: {(process_time*1000):.2f} ms")
  return res