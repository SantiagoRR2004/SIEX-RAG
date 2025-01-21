import downloader
import databaseCreator
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import sys
import os


def loadVectorStore(
    storeFolder: str = "./chroma_cve_db", embedder: str = "HuggingFaceEmbeddings"
) -> Chroma:
    """
    Load the vector store.

    Args:
        - storeFolder (str): The folder where the vector store is saved.
        - embedder (str): The embedder to use. It can be "HuggingFaceEmbeddings" or "OpenAIEmbeddings".

    Returns:
        - Chroma: The vector store.
    """
    if embedder == "OpenAIEmbeddings":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]
        )
    elif embedder == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"El embedder {embedder} no es válido")

    return Chroma(
        collection_name="cve_collection",
        embedding_function=embeddings,
        persist_directory=storeFolder,
    )


def createChatbot() -> StateGraph:
    """
    Create the chatbot.

    Returns:
        - StateGraph: The chatbot.
    """
    load_dotenv()  # carga OPENAI_API_KEY del fichero .env
    vector_store = loadVectorStore()

    llm_rag = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks in cybersecurity domains. "
                + "Use the provided pieces of retrieved context to give the MITRE technique and to give a solution. "
                + "If you don't know the answer, just say that you don't know. "
                + "Use three sentences maximum and keep the answer concise.",
            ),
            ("user", "Question: {question} \nContext: {context} \nAnswer:"),
        ]
    )

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieveContext(state: State, debug: bool = False) -> List[Document]:
        """
        Retrieve the context from the vector store.

        Args:
            - state (State): The state of the RAG.
            - debug (bool): If True, debug information will be printed.

        Returns:
            - List[Document]: The context.
        """
        retrieved_documents = vector_store.similarity_search(state["question"], k=5)
        if debug:
            print(
                "DEBUG::retrieveContext:: Num. documentos recuperados: {}".format(
                    len(retrieved_documents)
                )
            )
            print()
        return {"context": retrieved_documents}

    def generateAnswer(state: State, debug: bool = False) -> str:
        """
        Generate the answer to the question.

        Args:
            - state (State): The state of the RAG.
            - debug (bool): If True, debug information will be printed.

        Returns:
            - str: The answer.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )

        if debug:
            messages_text = str(messages)
            messages_start = messages_text[:150]
            messages_end = messages_text[len(messages_text) - 150 :]
            print(
                "DEBUG::generateAnswer:: Prompt: {} ... {}".format(
                    messages_start, messages_end
                )
            )
            print("DEBUG::generateAnswer::")
            print()

        response = llm_rag.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence(
        [
            ("retrieve", lambda state: retrieveContext(state, debug=True)),
            ("generate", lambda state: generateAnswer(state, debug=True)),
        ]
    )
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


def main():
    graph = createChatbot()
    print(
        "CHATBOT INICIADO.\nFinalizar sesión con los comandos :salir, :exit o :terminar"
    )
    while True:
        query = input(">> ")
        if query.lower() in [":salir", ":exit", ":terminar"]:
            sys.exit("Gracias por hablar conmigo!!!!")

        rag_response = graph.invoke({"question": query})
        print("[Chatbot con RAG] " + rag_response["answer"])
        print()


if __name__ == "__main__":
    downloader.downloadTechniquesEnterpriseAttack()
    databaseCreator.createVectorStore()
    main()
