"""Internal step — Compactor. Once a turn ends, collapses that turn's tool-call
chatter (retrieval call and retrieved documents) down to the final Q&A
pair plus a minimal marker of what was retrieved, so later turns don't keep
re-paying for it."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import RemoveMessage


def _retrieval_query(messages: list) -> str:
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] == "retrieve_documents":
                return call["args"].get("query", "")
    return ""


def _retrieval_summary(messages: list) -> tuple[str, list[str], bool]:
    query = _retrieval_query(messages)
    sources: list[str] = []
    failed = False
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        artifact = getattr(message, "artifact", None)
        if not isinstance(artifact, dict):
            continue
        failed = artifact.get("status") not in {None, "ok"}
        for source in artifact.get("sources", []):
            name = source.get("source") if isinstance(source, dict) else None
            if name and name not in sources:
                sources.append(name)
    return query, sources, failed


def make_compaction_node():
    def compaction_node(state: dict) -> dict:
        return
        messages = state["messages"]
        turn_start = next(
            i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)
        )
        chatter = messages[turn_start + 1 : -1]
        if not chatter:
            return {}

        query, sources, failed = _retrieval_summary(chatter)
        if failed:
            marker_content = f"[Retrieval failed for: {query}]"
        elif query:
            source_text = ", ".join(sources) if sources else "no sources returned"
            marker_content = f"[Retrieved for: {query} | Sources: {source_text}]"
        else:
            marker_content = "[Handled without retrieval]"
        marker = AIMessage(
            content=marker_content,
            name="retrieval",
            id=chatter[0].id,
        )
        removals = [RemoveMessage(id=message.id) for message in chatter[1:]]
        return {"messages": removals + [marker]}

    return compaction_node
