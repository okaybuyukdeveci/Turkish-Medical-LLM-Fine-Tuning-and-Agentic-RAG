"""The Responder (retrieval, chit-chat, and synthesis expert)."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from scripts.prompts.prompts import RESPONDER_SYSTEM_PROMPT_TEMPLATE

MAX_RETRIEVALS = 1


def _retrievals_this_turn(messages: list) -> int:
    """Count document retrievals since the last human message."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name == "retrieve_documents":
            count += 1
    return count


def make_responder_node(llm):
    """llm must already have `retrieve_documents` bound via .bind_tools()."""

    def responder_node(state: dict) -> dict:
        count = _retrievals_this_turn(state["messages"])
        system = SystemMessage(
            content=RESPONDER_SYSTEM_PROMPT_TEMPLATE.format(
                count=count, max_retrievals=MAX_RETRIEVALS
            )
        )
        messages = [system] + state["messages"]
        ai_message = llm.invoke(messages)
        ai_message.name = "responder"
        return {"messages": [ai_message]}

    return responder_node


def route_from_responder(state: dict) -> Literal["retrieval_tools", "compact"]:
    last_message = state["messages"][-1]
    if (
        getattr(last_message, "tool_calls", None)
        and _retrievals_this_turn(state["messages"]) < MAX_RETRIEVALS
    ):
        return "retrieval_tools"
    return "compact"
