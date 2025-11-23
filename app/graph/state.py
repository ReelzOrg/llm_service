import operator
import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Annotated, List, TypedDict, Optional, Literal, Dict, Any
from langchain_core.messages import BaseMessage

ModelProvider = Literal["ollama", "huggingface"]

# Gemma - Text, Image
# Qwen - Text
class ModelInfo(BaseModel):
  # provider: ModelProvider
  name: str
  parameters: Dict[str, Any] = {} # e.g. {"temperature": 0.7, "max_tokens": 512}

class GraphState(BaseModel):
  messages: Annotated[List[BaseMessage], operator.add] #The operator.add is used to chain the messages
  model_info: Optional[ModelInfo] = None
  user_id: Optional[str] = None
  
  timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
  session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  summary: Optional[str] = None 
  #The routing from 1 node to another depends on the output of the previous node hence we are not explicitly defining
  #the next_node key here