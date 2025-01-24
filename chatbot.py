import langchain_core.language_models
import langchain_core.vectorstores
import langchain_core.documents
import langchain_openai
from dotenv import load_dotenv
import os
import databaseCreator


class MITREATTACKChatbot:
    def __init__(self, verbose: bool = False, documentsInContext: int = 3) -> None:
        """
        Initialize the chatbot.

        Args:
            - verbose (bool): If True, debug information will be printed.
            - documentsInContext (int): The number of documents to retrieve.

        Returns:
            - None
        """
        self.verbose = verbose
        self.k = documentsInContext

    def main(self):
        pass

    def getLLMModel(self) -> langchain_core.language_models.BaseChatModel:
        """
        Get the large language model.

        Args:
            - None

        Returns:
            - BaseChatModel: The language model.
        """
        load_dotenv()  # carga OPENAI_API_KEY del fichero .env

        llm_rag = langchain_openai.ChatOpenAI(
            model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]
        )
        return llm_rag

    def getVectorStore(self) -> langchain_core.vectorstores.VectorStore:
        """
        Get the vector store.

        Args:
            - None

        Returns:
            - VectorStore: The vector store.
        """
        return databaseCreator.loadVectorStore()

    def buildGraph(self):
        pass

    def retrieveContext(self, query: str) -> list[langchain_core.documents.Document]:
        """
        Retrieve the context from the vector store.

        It also stores the documents it has retrieved
        in self.context.

        Args:
            - query (str): The question to answer.

        Returns:
            - list[Document]: The retrieved documents.
        """
        retrieved_docs = self.getVectorStore().similarity_search(query, k=self.k)

        if self.verbose:
            print(
                f"DEBUG::retrieveContext:: Num. documentos recuperados: {len(retrieved_docs)}"
            )

            for doc in retrieved_docs:
                print(f"DEBUG:: Doc: {doc.page_content[:100]}...")
            print()

        self.context = retrieved_docs

        return retrieved_docs

    def serializeContext(self):
        pass
