import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from middlewares.utils import calcTimeTaken
from utils.networkReq import close_session
from routes.llmRouter import router as chatRouter

@asynccontextmanager
async def lifespan(app: FastAPI):
  yield
  print("Shutting down......................................")
  await close_session()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["application/json", "application/x-www-form-urlencoded", "multipart/form-data"],
)
# app.middleware("http")(calcTimeTaken)
# app.add_middleware(HTTPSRedirectMiddleware)

app.mount("/public", StaticFiles(directory="public"), name="public")

app.include_router(chatRouter, prefix="/llm")

@app.get("/")
def root():
  return "Go to /llm/<user_id>/ask or /llm/<user_id>/chat to talk to the llm"