from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from scripts.workflow.graph import build_graph


class Retriever:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def invoke(self, query):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline")
        return [
            Document(
                page_content="# Başlık\n## Alt\n\nKanıt",
                metadata={
                    "source": "Dahiliye/a.md",
                    "title": "Başlık",
                    "subtitle": "Alt",
                    "chunk_index": 0,
                },
            )
        ]


class BoundResponder:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, messages):
        self.parent.bound_calls += 1
        if self.parent.chit_chat:
            return AIMessage(content="Merhaba")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "kalp"},
                    "id": f"call-{self.parent.bound_calls}",
                    "type": "tool_call",
                }
            ],
        )


class Responder:
    def __init__(self, chit_chat=False, failure_answer=False):
        self.chit_chat = chit_chat
        self.failure_answer = failure_answer
        self.bound_calls = 0
        self.unbound_calls = 0

    def bind_tools(self, tools):
        assert len(tools) == 1
        return BoundResponder(self)

    def invoke(self, messages):
        self.unbound_calls += 1
        content = "Kanıt bulunamadı" if self.failure_answer else "Kanıta dayalı cevap [Dahiliye/a.md]"
        return AIMessage(content=content)


class Pruner:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=messages[-1].content)


def test_graph_retrieves_once_prunes_and_compacts_sources():
    retriever = Retriever()
    responder = Responder()
    pruner = Pruner()
    graph = build_graph(responder, pruner, retriever)

    result = graph.invoke({"messages": [HumanMessage(content="Kalp nedir?")]})

    assert retriever.calls == 1
    assert pruner.calls == 1
    assert responder.bound_calls == 1
    assert responder.unbound_calls == 1
    assert result["messages"][-1].content.startswith("Kanıta dayalı")
    assert any("Sources: Dahiliye/a.md" in message.content for message in result["messages"])


def test_graph_skips_retrieval_for_chit_chat():
    retriever = Retriever()
    responder = Responder(chit_chat=True)
    graph = build_graph(responder, Pruner(), retriever)

    result = graph.invoke({"messages": [HumanMessage(content="Merhaba")]})

    assert result["messages"][-1].content == "Merhaba"
    assert retriever.calls == 0


def test_graph_reports_retrieval_failure_without_pruning():
    retriever = Retriever(fail=True)
    responder = Responder(failure_answer=True)
    pruner = Pruner()
    graph = build_graph(responder, pruner, retriever)

    result = graph.invoke({"messages": [HumanMessage(content="Bir bilgi ver")]})

    assert retriever.calls == 1
    assert pruner.calls == 0
    assert result["messages"][-1].content == "Kanıt bulunamadı"
    assert any("Retrieval failed" in message.content for message in result["messages"])


def test_graph_accepts_a_second_conversation_turn():
    retriever = Retriever()
    responder = Responder()
    graph = build_graph(responder, Pruner(), retriever)
    first = graph.invoke({"messages": [HumanMessage(content="Kalp nedir?")]})

    second = graph.invoke(
        {
            "messages": first["messages"]
            + [HumanMessage(content="Peki tedavisi nedir?")]
        }
    )

    assert retriever.calls == 2
    assert second["messages"][-1].content.startswith("Kanıta dayalı")
    assert sum(isinstance(message, HumanMessage) for message in second["messages"]) == 2
