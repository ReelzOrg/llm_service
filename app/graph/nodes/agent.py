from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline

from ..state import GraphState, ModelProvider
from ..tools.search import searxng_search
from ..prompts import agent_prompt_template

def get_model(provider: ModelProvider, model_name: str):
  if provider == "ollama":
    return ChatOllama(model=model_name, temperature=0.5)

  if provider == "huggingface":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)  # Use bfloat16 for memory efficiency
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=hf_pipeline)

  raise ValueError(f"Unknown provider {provider}")

def agent_node(state: GraphState):
  provider = state["model_provider"]
  model_name = state["model_name"]
  print(f"---NODE: AGENT (PROVIDER: {provider}, MODEL: {model_name})---")

  llm = get_model(provider, model_name)
  llm_with_tools = llm.bind_tools([searxng_search])
  chain = agent_prompt_template | llm_with_tools

  response = chain.invoke({"messages": state["messages"]})

  return {"messages": [response]}