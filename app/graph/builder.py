from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes.agent import agent_node
from .nodes.triage import triage_node
from .tools.search import searxng_search
from .state import GraphState

def should_continue(state: GraphState):
  if state.messages[-1].tool_calls:
    return "tools"

  # End the conversation/graph
  return END

def build_graph():
  tool_node = ToolNode([searxng_search])

  workflow = StateGraph(GraphState)
  workflow.add_node("triage", triage_node)
  workflow.add_node("agent", agent_node)
  workflow.add_node("tools", tool_node)

  workflow.set_entry_point("triage")
  workflow.add_edge("triage", "agent")

  workflow.add_conditional_edges("agent", should_continue, {
    # The key ("tools") must match the name we gave the ToolNode.
    "tools": "tools",
    END: END
  })

  # After the tools are executed, the flow goes back to the agent
  # to process the tool's output.
  workflow.add_edge("tools", "agent")

  return workflow.compile(checkpointer=MemorySaver())

graph = build_graph()