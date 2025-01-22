import databaseCreator
from langgraph.graph import START, StateGraph, MessagesState, END
from typing_extensions import List, TypedDict
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import sys
import os


# codigo basado en: https://python.langchain.com/docs/tutorials/qa_chat_history/
def createChatbot() -> StateGraph:
    """
    Create the chatbot.

    Returns:
        - StateGraph: The chatbot.
    """
    load_dotenv()  # carga OPENAI_API_KEY del fichero .env
    vector_store = databaseCreator.loadVectorStore()

    llm_rag = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    graph_builder = StateGraph(MessagesState)

    @tool(response_format="content_and_artifact")
    def retrieveContext(query: str, debug: bool = True):
        """
        Retrieve the context from the vector store.

        Args:
            - query (str): The question to answer.
            - debug (bool): If True, debug information will be printed.
        """
        retrieved_docs = vector_store.similarity_search(query, k=3)
        if debug:
            print(
                "DEBUG::retrieveContext:: Num. documentos recuperados: {}".format(
                    len(retrieved_docs)
                )
            )
            for doc in retrieved_docs:
                print(f"DEBUG:: Doc: {doc.page_content[:100]}...")
            print()
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm_rag.bind_tools([retrieveContext])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieveContext])

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks in cybersecurity domains. "
            "Use the provided pieces of retrieved context to give the MITRE technique and a solution. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "Then, answer questions and clarifications regarding your proposed answer/solution"
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm_rag.invoke(prompt)
        return {"messages": [response]}

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # from PIL import Image
    # from io import BytesIO
    # image = Image.open(BytesIO(graph.get_graph().draw_mermaid_png()))
    # image.show()

    return graph


def main():
    graph = createChatbot()
    config = {"configurable": {"thread_id": "1111"}}
    print(
        "CHATBOT INICIADO.\nFinalizar sesión con los comandos :salir, :exit o :terminar"
    )
    while True:
        query = input(">> ")
        if query.lower() in [":salir", ":exit", ":terminar"]:
            sys.exit("Gracias por hablar conmigo!!!!")

        for step in graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()
        print()


if __name__ == "__main__":
    main()
