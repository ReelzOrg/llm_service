from langgraph.checkpointer.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage

async def save_chat_history(thread_id: str, message: list[HumanMessage | AIMessage]):
	saver = AsyncPostgresSaver(
		connection_string="postgresql+asyncpg://postgres:Vivek2002&@localhost:5432/llmchats",
		table_name="chat_history"
	)
	await saver.aput(thread_id, message)