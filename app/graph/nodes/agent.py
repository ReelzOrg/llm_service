from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

from ..state import ChatGraphState
from ..tools.search import searxng_search
from ..prompts import agent_prompt_template

def make_agent_node(model: ChatOllama, tools: list[BaseTool], sysPrompt: ChatPromptTemplate):
	model_with_tools = model.bind_tools(tools)
	agent = sysPrompt | model_with_tools

	# check if the messages fit within the model's context length
	# model.get_num_tokens(input_prompt)

	tokenizer = AutoTokenizer.from_pretrained(model.model)
	len(tokenizer.encode())

	def agent_node(state: ChatGraphState):
		return {"messages": [agent.invoke({"messages": state["messages"]})]}

	return agent_node

# Fast Model
# 256k, text + image, vision + tools
fastModelNode = make_agent_node(
	ChatOllama(model="qwen3-vl:2b", temperature=0),
	[searxng_search],
	agent_prompt_template
)

# Complex Model
# 256k, text + image, vision + tools
complexModelNode = make_agent_node(
	ChatOllama(model="qwen3-vl:4b", temperature=0),
	[searxng_search],
	agent_prompt_template
)

# Coding Model
# 32k, text, tools
codingModelNode = make_agent_node(
	ChatOllama(model="qwen2.5-coder:0.5b", temperature=0),
	[searxng_search],
	agent_prompt_template
)