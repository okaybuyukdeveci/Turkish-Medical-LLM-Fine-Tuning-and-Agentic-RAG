"""Internal step — Compactor. Once a turn ends, collapses that turn's tool-call
chatter (retrieval call and retrieved documents) down to the final Q&A
pair plus a minimal marker of what was retrieved, so later turns don't keep
re-paying for it."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import RemoveMessage


def _retrieval_query(messages: list) -> str:
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] == "retrieve_documents":
                return call["args"].get("query", "")
    return ""


def make_compaction_node():
    def compaction_node(state: dict) -> dict:
        messages = state["messages"]
        turn_start = next(
            i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)
        )
        chatter = messages[turn_start + 1 : -1]
        if not chatter:
            return {}

        query = _retrieval_query(chatter)
        marker = AIMessage(
            content=f"[Retrieved documents for: {query}]" if query else "[Handled without retrieval]",
            name="retrieval",
            id=chatter[0].id,
        )
        removals = [RemoveMessage(id=message.id) for message in chatter[1:]]
        return {"messages": removals + [marker]}

    return compaction_node
