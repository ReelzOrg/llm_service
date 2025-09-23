from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from ..state import GraphState, ModelProvider
from ..prompts import triage_prompt

# Right now we are using a smaller cheap model for dynamic routing to more specialized LLM.
# I will add XGBoost for classification and routing
triage_llm = ChatOllama(model="gemma3:1b", temperature=0)

class Triage(BaseModel):
  model_provider: ModelProvider = Field(description="The model provider to use for the next node")
  model_name: str = Field(description="The name of the model to use from the model_provider")

# Configuring the LLM to force its output into the `Triage` schema.
structured_llm_triage = triage_llm.with_structured_output(Triage)
triage_chain = triage_prompt | structured_llm_triage

def triage_node(state: GraphState):
  prompt = state["messages"][-1].content
  result = triage_chain.invoke({"prompt": prompt})

  # We dont need to update the state here as LangGraph will do it for us
  # state["model_provider"] = result.model_provider
  # state["model_name"] = result.model_name
  print(f"---ROUTING TO PROVIDER: {result.model_provider}, MODEL: {result.model_name}---")
  return {
    "model_provider": result.model_provider,
    "model_name": result.model_name
  }