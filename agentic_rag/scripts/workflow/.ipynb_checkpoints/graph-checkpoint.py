"""LangGraph workflow wiring the Responder, Retriever, and Pruner together."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from scripts.agents.pruning_agent import make_pruning_node
from scripts.agents.responder_agent import make_responder_node, route_from_responder
from scripts.agents.retriever_agent import make_retriever_node, route_from_retriever
from scripts.functions.tools import retrieve_documents, transfer_to_retriever


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(responder_llm, retriever_llm, pruning_llm):
    """responder_llm must have `transfer_to_retriever` bound; retriever_llm must have
    `retrieve_documents` bound (both via .bind_tools()); pruning_llm needs no tools."""
    graph = StateGraph(GraphState)

    graph.add_node("responder", make_responder_node(responder_llm))
    graph.add_node("retriever", make_retriever_node(retriever_llm))
    graph.add_node("responder_tools", ToolNode([transfer_to_retriever]))
    graph.add_node("retriever_tools", ToolNode([retrieve_documents]))
    graph.add_node("prune", make_pruning_node(pruning_llm))

    graph.add_edge(START, "responder")
    graph.add_conditional_edges("responder", route_from_responder, ["responder_tools", END])
    graph.add_conditional_edges("retriever", route_from_retriever, ["retriever_tools", "responder"])
    graph.add_edge("responder_tools", "retriever")
    graph.add_edge("retriever_tools", "prune")
    graph.add_edge("prune", "retriever")

    return graph.compile()
