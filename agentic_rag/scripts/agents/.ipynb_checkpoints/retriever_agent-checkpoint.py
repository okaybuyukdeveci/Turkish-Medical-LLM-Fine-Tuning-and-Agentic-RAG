"""Agent 1 — The Retriever (document & query expert)."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from scripts.prompts.prompts import RETRIEVER_SYSTEM_PROMPT_TEMPLATE

MAX_RETRIEVALS = 3


def _retrievals_this_turn(messages: list) -> int:
    """Count retrieve_documents results since the last human message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name == "retrieve_documents":
            count += 1
    return count


def make_retriever_node(llm):
    """llm must already have `retrieve_documents` bound via .bind_tools()."""

    def retriever_node(state: dict) -> dict:
        count = _retrievals_this_turn(state["messages"])
        system = SystemMessage(
            content=RETRIEVER_SYSTEM_PROMPT_TEMPLATE.format(count=count, max_retrievals=MAX_RETRIEVALS)
        )
        messages = [system] + state["messages"]
        ai_message = llm.invoke(messages)
        ai_message.name = "retriever"
        if not ai_message.tool_calls:
            ai_message.content = ""
        return {"messages": [ai_message]}

    return retriever_node


def route_from_retriever(state: dict) -> Literal["retriever_tools", "responder"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None) and _retrievals_this_turn(state["messages"]) < MAX_RETRIEVALS:
        return "retriever_tools"
    return "responder"
