from app.graph.state import GraphState

from app.utils.db.connectQdrant import vectorStore

async def retrieve_memory(state: GraphState):
	"""Query Qdrant for past context relevant to the user's LATEST message."""
	latest_msg = state["messages"][-1].content

	# Search for similar past messages (e.g., top 3)
	results = await vectorStore.asimilarity_search(latest_msg, k=3)