import langchain_core.language_models
import langchain_core.vectorstores
import langchain_core.documents
import langchain_openai
import langchain_core.messages
from dotenv import load_dotenv
import os
import databaseCreator
import sys


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
        self.model = self.getLLMModel()
        self.messages = [self.getInitialPrompt()]

    def main(self):
        print(
            "CHATBOT INICIADO.\nFinalizar sesión con los comandos :salir, :exit o :terminar"
        )
        while True:
            query = input(">> ")
            if query.lower() in [":salir", ":exit", ":terminar"]:
                sys.exit("Gracias por hablar conmigo!!!!")

            if query.lower() == ":reset":
                pass

            # If there is only one message we retrieve the context
            if len(self.messages) == 1:
                self.retrieveContext(query)

                # # Chec if there is a way to make it work with ToolMessage
                # self.messages.append(
                #     langchain_core.messages.ToolMessage(
                #         self.serializeContext(), tool_call_id="unique_tool_call_id_123"
                #     )
                # )
                self.messages.append(
                    langchain_core.messages.SystemMessage(self.serializeContext())
                )

            self.messages.append(langchain_core.messages.HumanMessage(content=query))

            response = self.callModel()

            print(response.content)

            # We add the response to the messages so it has memory
            self.messages.append(response)

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


if __name__ == "__main__":
    chatbot = MITREATTACKChatbot()
    chatbot.main()
