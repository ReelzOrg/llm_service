from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from typing import Optional
from ..state import GraphState, ModelProvider
from ..prompts import triage_prompt

# Right now we are using a smaller cheap model for dynamic routing to more specialized LLM.
# I will add XGBoost for classification and routing
# triage_llm = ChatOllama(model="gemma3:1b", temperature=0)

# class Triage(BaseModel):
#   model_provider: Optional[ModelProvider] = Field(description="The model provider to use for the next node")
#   model_name: Optional[str] = Field(description="The name of the model to use from the model_provider")

# # Configuring the LLM to force its output into the `Triage` schema.
# structured_llm_triage = triage_llm.with_structured_output(Triage)
# triage_chain = triage_prompt | structured_llm_triage

def triage_node(state: GraphState):
  # If model is already specified in the state (from request), use it.
  if state.model_info and state.model_info.name:
    print(f"---USING REQUESTED MODEL: {state.model_info.name}---")
    return {}

  # Otherwise, default to a model (since the dynamic routing logic is currently disabled)
  print(f"---ROUTING TO DEFAULT MODEL: qwen3:4b---")
  return {
    "model_info": {
      "provider": "ollama",
      "name": "qwen3:4b"
    }
  }