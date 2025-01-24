import langchain_core.language_models
import langchain_core.vectorstores
import langchain_openai
from dotenv import load_dotenv
import os
import databaseCreator


class MITREATTACKChatbot:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        pass

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
