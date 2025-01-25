import langchain_core.language_models
import langchain_core.vectorstores
import langchain_core.documents
import langchain_core.messages
from abc import ABC, abstractmethod


class Chatbot(ABC):
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
        self.model = self.getLLMModel()
        self.resetMemory()
        self.vectorStore = self.getVectorStore()

    def main(self):
        print(
            "CHATBOT INICIADO.\n"
            + "Finalizar sesión con los comandos :salir, :exit o :terminar"
        )
        while True:
            query = self.getUserInput()
            if query.lower() in [":salir", ":exit", ":terminar"]:
                print("Gracias por hablar conmigo!!!!")
                break

            if query.lower() == ":reset":
                self.resetMemory()

            else:

                # If there is only one message we retrieve the context
                if len(self.messages) == 1:
                    self.retrieveContext(query)

                    # # Check if there is a way to make it work with ToolMessage
                    # self.messages.append(
                    #     langchain_core.messages.ToolMessage(
                    #         self.serializeContext(), tool_call_id="unique_tool_call_id_123"
                    #     )
                    # )
                    self.messages.append(
                        langchain_core.messages.SystemMessage(self.serializeContext())
                    )

                self.messages.append(
                    langchain_core.messages.HumanMessage(content=query)
                )

                response = self.callModel()

                response.pretty_print()

                # We add the response to the messages so it has memory
                self.messages.append(response)

    @abstractmethod
    def getLLMModel(self) -> langchain_core.language_models.BaseChatModel:
        """
        Get the large language model.

        Args:
            - None

        Returns:
            - BaseChatModel: The language model.
        """
        pass

    @abstractmethod
    def getVectorStore(self) -> langchain_core.vectorstores.VectorStore:
        """
        Get the vector store.

        Args:
            - None

        Returns:
            - VectorStore: The vector store.
        """
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
        retrieved_docs = self.vectorStore.similarity_search(query, k=self.k)

        if self.verbose:
            print(
                f"DEBUG::retrieveContext:: Num. documentos recuperados: {len(retrieved_docs)}"
            )

            for doc in retrieved_docs:
                print(f"DEBUG:: Doc: {doc.page_content[:100]}...")
            print()

        self.context = retrieved_docs

        return retrieved_docs

    def serializeContext(self) -> str:
        """
        Serialize the context.

        It returns a string with
        the metadata and the content of the documents.

        The metadata had the folowing keys:
            {id, url, seq_num, source}

        Later we could eliminate the source attribute
        because we only have on json.

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

            serializedDocuments.append(
                f"Source: {metadata}\nContent: {doc.page_content}"
            )

            if set(metadata.keys()) not in metadataKeys:
                metadataKeys.append(set(metadata.keys()))

        if self.verbose:
            for keys in metadataKeys:
                print(keys)
            print()

        return "\n\n".join(serializedDocuments)

    @abstractmethod
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
        pass

    def callModel(self) -> langchain_core.messages.AIMessage:
        """
        Call the model.

        It generates the response to the user
        using the model.

        Args:
            - None

        Returns:
            - AIMessage: The response to the user.
        """
        response = self.model.invoke(self.messages)

        self.lastResponse = response

        return response

    def resetMemory(self) -> None:
        """
        Reset the memory of the chatbot.

        Args:
            - None

        Returns:
            - None
        """

        if getattr(self, "messages", None) is not None:
            print("\nMemoria reseteada.\n")

        self.messages = [self.getInitialPrompt()]
        self.context = []
        self.lastResponse = None

    @abstractmethod
    def getUserInput(self) -> str:
        """
        Get the user input.

        Args:
            - None

        Returns:
            - str: The user input.
        """
        pass
