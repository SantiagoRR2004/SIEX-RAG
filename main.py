import databaseCreator
from typing import Sequence
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Annotated
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import random
import sys
import os


def generate_tread_id(excluded_ids: List[str] = []) -> str:
    """
    Generate a thread id.

    Args:
        - excluded_ids (List[str]): The list of excluded thread ids.

    Returns:
        - str: The generated thread id.
    """
    while True:
        random_number = random.randint(1000, 9999)
        if random_number not in excluded_ids:
            return str(random_number)


def createChatbot(debug: bool = False) -> StateGraph:
    """
    Create the chatbot.

    Args:
        - debug (bool): If True, debug information will be printed.

    Returns:
        - StateGraph: The chatbot.
    """
    load_dotenv()  # carga OPENAI_API_KEY del fichero .env
    vector_store = databaseCreator.loadVectorStore()

    llm_rag = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks in cybersecurity domains. "
                + "Use the provided pieces of retrieved context to give the MITRE technique and a solution. "
                + "If you don't know the answer, just say that you don't know. "
                + "Use three sentences maximum and keep the answer concise."
                + "Then, answer questions and clarifications regarding your proposed answer/solution",
            ),
            # We will utilize MessagesPlaceholder to pass all the messages in.
            MessagesPlaceholder(variable_name="messages"),
            ("user", "Question: {question} \nContext: {context}"),
        ]
    )

    class State(TypedDict):
        # The messages in the conversation.
        messages: Annotated[Sequence[BaseMessage], add_messages]
        context: List[Document]  # The context to answer the question.

    def retrieveContext(state: State, debug: bool = False) -> State:
        """
        Retrieve the context from the vector store.

        Args:
            - state (State): The state of the RAG.
            - debug (bool): If True, debug information will be printed.

        Returns:
            - State: The state with the retrieved context.
        """
        # Extract the query from the last message.
        query = state["messages"][-1].content
        retrieved_documents = vector_store.similarity_search(query, k=3)
        if debug:
            print(
                "DEBUG::retrieveContext:: Num. documentos recuperados: {}".format(
                    len(retrieved_documents)
                )
            )
            for doc in retrieved_documents:
                print(f"DEBUG:: Doc: {doc.page_content[:100]}...")
            print()

        # We will pass the retrieved documents as context to the next step.
        return {"context": retrieved_documents}

    def generateAnswer(state: State, debug: bool = False) -> State:
        """
        Generate the answer to the question.

        Args:
            - state (State): The state of the RAG.
            - debug (bool): If True, debug information will be printed.

        Returns:
            - State: The state with the answer.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        query = state["messages"][-1].content
        # Invoke the RAG prompt.
        # We pass the messages (except the last one), the question and the context.
        messages = rag_prompt.invoke(
            {
                "messages": state["messages"][:-1],
                "question": query,
                "context": docs_content,
            }
        )

        if debug:
            messages_text = str(messages)
            messages_start = messages_text[:150]
            messages_end = messages_text[len(messages_text) - 150 :]
            # print(
            #     "DEBUG::generateAnswer:: Prompt: {} ... {}".format(
            #         messages_start, messages_end
            #     )
            # )
            print("DEBUG::generateAnswer::" + messages_text)
            print("DEBUG::generateAnswer::")
            print()

        response = llm_rag.invoke(messages)
        return {"messages": response}

    graph_builder = StateGraph(State).add_sequence(
        [
            ("retrieve", lambda state: retrieveContext(state, debug=debug)),
            ("generate", lambda state: generateAnswer(state, debug=debug)),
        ]
    )
    graph_builder.add_edge(START, "retrieve")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def main():
    graph = createChatbot(debug=True)
    thread_id = generate_tread_id()
    excluded_ids = [thread_id]
    config = {"configurable": {"thread_id": thread_id}}
    print(
        "CHATBOT INICIADO.\nFinalizar sesión con los comandos :salir, :exit o :terminar"
    )
    while True:
        query = input(">> ")
        if query.lower() in [":salir", ":exit", ":terminar"]:
            sys.exit("Gracias por hablar conmigo!!!!")

        if query.lower() == ":reset":
            thread_id = generate_tread_id(excluded_ids)
            config = {"configurable": {"thread_id": thread_id}}
            excluded_ids.append(thread_id)
        else:
            query_message = [HumanMessage(query)]
            rag_response = graph.invoke({"messages": query_message}, config)
            rag_response["messages"][-1].pretty_print()
            print()


if __name__ == "__main__":
    main()
