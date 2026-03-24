# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, StateGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from .nodes.agent import complexModelNode, codingModelNode, fastModelNode
# from .nodes.triage import triage_node
from .router import router_node
from .tools.search import searxng_search
from .state import ChatGraphState

def should_continue(state: ChatGraphState):
  last_message = state["messages"][-1]

  # Check if it is an AIMessage first because tool_calls is only available on AIMessage
  if isinstance(last_message, AIMessage) and last_message.tool_calls:
    return "tools"
  return END

# Right now we are looping back to the same model selected by the router but in future we can perform some more operations
# and based on the output of the tool as well we can redirect the flow to some other model
def route_back_to_agent(state: ChatGraphState):
  if state["model_info"]:
    return state["model_info"]["name"]
  return END

def route_logic(state: ChatGraphState):
  if state["model_info"]:
    return state["model_info"]["name"]
  return END

def build_graph(checkpointer):
  tool_node = ToolNode([searxng_search])

  workflow = StateGraph(ChatGraphState)
  workflow.add_node("router", router_node)
  workflow.add_node("complex_model", complexModelNode)
  workflow.add_node("coding_model", codingModelNode)
  workflow.add_node("fast_model", fastModelNode)

  # workflow.add_node("triage", triage_node)
  # workflow.add_node("agent", agent_node)
  workflow.add_node("tools", tool_node)

  workflow.set_entry_point("router")
  # Edge 1: Router -> SpecificModel
  workflow.add_conditional_edges("router", route_logic, {"complex_model": "complex_model", "coding_model": "coding_model", "fast_model": "fast_model"})

  # Edge 2: Model -> Tools (or END)
  # We apply this logic to ALL three models
  for node_name in ["fast_model", "complex_model", "coding_model"]:
    workflow.add_conditional_edges(node_name, should_continue, {"tools": "tools", END: END})

  # Edge 3: Tools -> Back to Agent
  # This completes the loop
  workflow.add_conditional_edges(
    "tools",
    route_back_to_agent,
    {"fast_model": "fast_model","complex_model": "complex_model","coding_model": "coding_model"}
  )
  
  # workflow.add_edge("triage", "agent")
  # workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

  # After the tools are executed, the flow goes back to the agent
  # to process the tool's output.
  # workflow.add_edge("tools", "agent")

  return workflow.compile(checkpointer=checkpointer)

# graph = build_graph(MemorySaver())