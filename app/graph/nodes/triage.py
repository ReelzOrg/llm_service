from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from ..state import GraphState, ModelProvider

# Right now we are using a smaller cheap model for dynamic routing to more specialized LLM, but maybe creating a
# custom model is faster
triage_llm = ChatOllama(model="gemma3:1b", temperature=0)

class Triage(BaseModel):
  model_provider: ModelProvider = Field(description="The model provider to use for the next node")
  model_name: str = Field(description="The name of the model to use from the model_provider")

# Configuring the LLM to force its output into the `Triage` schema.
structured_llm_triage = triage_llm.with_structured_output(Triage)

triage_prompt = ChatPromptTemplate.from_messages([
  ("system", """You are an expert at routing a user's request to the correct model provider and model.
  Based on the user's prompt, decide which provider and model to use.
  
  - For general conversation or simple tasks that can run locally, use 'ollama' with the 'gemma3:4b' model.
  - For coding specific tasks, use 'ollama' with the 'qwen2.5-coder:7b' model.
  """),
  ("human", "User prompt: {prompt}")
])

triage_chain = triage_prompt | structured_llm_triage

def triage_node(state: GraphState):
  prompt = state["messages"][-1].content
  result = triage_chain.invoke({"prompt": prompt})
  # state["model_provider"] = result.model_provider
  # state["model_name"] = result.model_name
  print(f"---ROUTING TO PROVIDER: {result.model_provider}, MODEL: {result.model_name}---")
  return {
    "model_provider": result.model_provider,
    "model_name": result.model_name
  }