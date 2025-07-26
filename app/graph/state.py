import operator
from typing import Annotated, List, TypedDict, Literal
from langchain_core.messages import BaseMessage

# Gemma - Text, Image
# Qwen - Text
ModelProvider = Literal["ollama", "huggingface"]

class GraphState(TypedDict):
  messages: Annotated[List[BaseMessage], operator.add] #The operator.add is used to chain the messages
  model_provider: ModelProvider
  user_id: str
  #The routing from 1 node to another depends on the output of the previous node hence we are not explicitly defining
  #the next_node key here