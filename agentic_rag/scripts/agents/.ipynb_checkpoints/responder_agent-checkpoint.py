"""Agent 2 — The Responder (chit-chat & synthesis expert)."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import END


from scripts.prompts.prompts import RESPONDER_SYSTEM_PROMPT_TEMPLATE

MAX_TRANSFERS = 1


def _transfers_this_turn(messages: list) -> int:
    """Count transfer_to_retriever handoffs since the last human message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name == "transfer_to_retriever":
            count += 1
    return count


def make_responder_node(llm):
    """llm must already have `transfer_to_retriever` bound via .bind_tools()."""

    def responder_node(state: dict) -> dict:
        count = _transfers_this_turn(state["messages"])
        system = SystemMessage(
            content=RESPONDER_SYSTEM_PROMPT_TEMPLATE.format(count=count, max_transfers=MAX_TRANSFERS)
        )
        messages = [system] + state["messages"]
        ai_message = llm.invoke(messages)
        ai_message.name = "responder"
        return {"messages": [ai_message]}

    return responder_node


def route_from_responder(state: dict) -> Literal["responder_tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None) and _transfers_this_turn(state["messages"]) < MAX_TRANSFERS:
        return "responder_tools"
    return END

