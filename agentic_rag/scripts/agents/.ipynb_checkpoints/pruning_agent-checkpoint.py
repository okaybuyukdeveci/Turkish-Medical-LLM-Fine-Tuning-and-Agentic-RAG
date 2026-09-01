"""Internal step — Pruner. Extracts only the part of a freshly retrieved document
that answers the Retriever's search query, before the rest enters shared context."""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from scripts.prompts.prompts import PRUNING_SYSTEM_PROMPT_TEMPLATE


def _last_retrieval_query(messages: list) -> str:
    tool_message = messages[-1]
    preceding_ai = messages[-2]
    for call in preceding_ai.tool_calls:
        if call["id"] == tool_message.tool_call_id:
            return call["args"]["query"]
    return ""


def make_pruning_node(llm):
    def pruning_node(state: dict) -> dict:
        tool_message = state["messages"][-1]
        query = _last_retrieval_query(state["messages"])
        system = SystemMessage(
            content=PRUNING_SYSTEM_PROMPT_TEMPLATE.format(initial_request=query)
        )
        pruned = llm.invoke([system, HumanMessage(content=tool_message.content)])
        updated_tool_message = ToolMessage(
            content=pruned.content,
            name=tool_message.name,
            tool_call_id=tool_message.tool_call_id,
            id=tool_message.id,
        )
        return {"messages": [updated_tool_message]}

    return pruning_node
