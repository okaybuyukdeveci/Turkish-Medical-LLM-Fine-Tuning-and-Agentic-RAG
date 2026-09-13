"""LangGraph workflow wiring the Responder, retrieval tool, and Pruner together."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from scripts.agents.compaction_agent import make_compaction_node
from scripts.agents.pruning_agent import make_pruning_node
from scripts.agents.responder_agent import make_responder_node, route_from_responder
from scripts.tools.retrieval_tool import create_retrieve_documents_tool


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(responder_llm, pruning_llm, retriever):
    """Build the graph and bind one shared retrieval-tool instance internally."""
    graph = StateGraph(GraphState)
    retrieval_tool = create_retrieve_documents_tool(retriever)

    graph.add_node("responder", make_responder_node(responder_llm, retrieval_tool))
    graph.add_node("retrieval_tools", ToolNode([retrieval_tool]))
    graph.add_node("prune", make_pruning_node(pruning_llm))
    graph.add_node("compact", make_compaction_node())

    graph.add_edge(START, "responder")
    graph.add_conditional_edges("responder", route_from_responder, ["retrieval_tools", "compact"])
    graph.add_edge("retrieval_tools", "prune")
    graph.add_edge("prune", "responder")
    graph.add_edge("compact", END)

    return graph.compile()
