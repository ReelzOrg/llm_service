from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from .state import GraphState
from typing import Literal, cast

class RouteQuery(BaseModel):
	"""Route the user query to the most relevant model."""
	destination_llm: Literal["complex_model", "fast_model", "coding_model"] = Field(description="The model to use based on query complexity.")

# Right now we are using a smaller cheap model for dynamic routing to more specialized LLM.
# I will add XGBoost for classification and routing
# Runnable[LanguageModelInput, dict | BaseModel]
router_llm = ChatOllama(model="gemma3:1b", temperature=0)
structured_router = router_llm.with_structured_output(RouteQuery)

def router_node(state: GraphState) -> dict[str, object]:
	# If the model was selected by the user then simply return it, we dont have to route it 
	if state["model_info"]:
		return {"model_info": state["model_info"]}
	
	last_message = state["messages"][-1]

	# Right now the llm is only used for generating text so str should be enough and simpler
	# but later if we are generating images or videos or audio then the dict[str, object] will be needed
	cleaned_content = cast(str | list[str | dict[str, object]], last_message.content)

	# We were first using tuples which only works if the content is a string
	# In HumanMessage and SystemMessage we can pass objects (which may be images, videos, audio, etc.)
	result = structured_router.invoke([
		SystemMessage(content="You are a router. Analyze the query. If it requires heavy reasoning or math, choose 'complex_model'. If it is code, choose 'coding_model'. Otherwise for simple chats use 'fast_model'."),
		HumanMessage(content=cleaned_content)
	])

	# BasedPyright doesn't know that the result is a RouteQuery so we need to cast it
	result = cast(RouteQuery, result)
	print("------------------ROUTER RESULT------------------------\n", result)
	print("------------------ROUTER RESULT END------------------------")
	return {"model_info": {"name": result.destination_llm}}