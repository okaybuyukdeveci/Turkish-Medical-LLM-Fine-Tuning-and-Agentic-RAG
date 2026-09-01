"""LangGraph workflow wiring the Responder, retrieval tool, and Pruner together."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from scripts.agents.compaction_agent import make_compaction_node
from scripts.agents.pruning_agent import make_pruning_node
from scripts.agents.responder_agent import make_responder_node, route_from_responder
from scripts.functions.tools import create_retrieve_documents_tool


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(responder_llm, pruning_llm, compressed_retriever):
    """responder_llm must have `retrieve_documents` bound; pruning_llm needs no tools."""
    graph = StateGraph(GraphState)

    graph.add_node("responder", make_responder_node(responder_llm))
    graph.add_node("retrieval_tools", ToolNode([create_retrieve_documents_tool(compressed_retriever)]))
    graph.add_node("prune", make_pruning_node(pruning_llm))
    graph.add_node("compact", make_compaction_node())

    graph.add_edge(START, "responder")
    graph.add_conditional_edges("responder", route_from_responder, ["retrieval_tools", "compact"])
    graph.add_edge("retrieval_tools", "prune")
    graph.add_edge("prune", "responder")
    graph.add_edge("compact", END)

    return graph.compile()
