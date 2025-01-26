import chatbot
import langchain_core.language_models
import langchain_core.vectorstores
import langchain_core.documents
import langchain_openai
import langchain_core.messages
from dotenv import load_dotenv
import os
import databaseCreator


class MITREATTACKChatbot(chatbot.Chatbot):
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

    def getInitialPrompt(self) -> langchain_core.messages.SystemMessage:
        """
        Get the initial prompt.

        It is a message that explains to
        the chatbot what it has to do.

        Args:
            - None

        Returns:
            - SystemMessage: The initial prompt.
        """
        message = (
            "You are an assistant for question-answering tasks in cybersecurity domains. "
            + "Use the provided pieces of retrieved context to give the MITRE technique and a solution. "
            + "If you don't know the answer, just say that you don't know. "
            + "Use three sentences maximum and keep the answer concise."
            + "Then, answer questions and clarifications regarding your proposed answer/solution"
        )

        return langchain_core.messages.SystemMessage(content=message)

    def getUserInput(self) -> str:
        """
        Get the user input.

        Args:
            - None

        Returns:
            - str: The user input.
        """
        return input(">> ")

    def serializeContext(self) -> str:
        """
        Serialize the context.

        It returns a string with
        the metadata and the content of the documents.

        The metadata is a dictionary that requires
        the following keys:
            - id: The ID of the MITRE technique.
            - url: The URL of the MITRE technique.
            - name: The name of the MITRE technique.
            - tactics: The tactics used by the MITRE technique.
            - platforms: The platforms affected by the MITRE technique.

        The content is a string.

        Args:
            - None

        Returns:
            - str: The serialized context.
        """
        serializedDocuments = []

        metadataKeys = []

        for doc in self.context:

            metadata = doc.metadata

            serializedDocument = f"Content: {doc.page_content}\n"

            serializedDocument += f"The ID, URL and name of the MITRE technique are {metadata['id']}, {metadata['url']} and {metadata['name']}, respectively."
            serializedDocument += f"The tactics and platforms affected are {metadata['tactics']} and {metadata['platforms']}."

            serializedDocuments.append(serializedDocument)

            if set(metadata.keys()) not in metadataKeys:
                metadataKeys.append(set(metadata.keys()))

        if self.verbose:
            for keys in metadataKeys:
                print(keys)
            print()

        return "\n\n".join(serializedDocuments)


if __name__ == "__main__":
    chat = MITREATTACKChatbot(verbose=True)
    chat.main()
