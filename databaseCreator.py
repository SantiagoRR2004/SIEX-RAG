from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pickle
import downloader


def createVectorStore(
    *,
    destitationFolder: str = "./chroma_cve_db",
    createAgain: bool = False,
    embedder: str = "HuggingFaceEmbeddings",
) -> None:
    """
    Create a vector store with the CVEs.

    Args:
        - destitationFolder (str): The folder where the vector store will be saved.
        - createAgain (bool): If True, the vector store will be created again,
            even if it already exists.
        - embedder (str): The embedder to use. It can be "HuggingFaceEmbeddings" or "OpenAIEmbeddings".

    Returns:
        - None
    """
    if (
        os.path.exists(destitationFolder)
        and os.path.exists("embeddings.pkl")
        and not createAgain
    ):
        print(f"\t- indice {destitationFolder} ya existe")

    else:
        folder = os.path.dirname(os.path.abspath(__file__))
        downloader.downloadTechniquesEnterpriseAttack()

        print("Cargando documentos ...")
        loader = JSONLoader(
            file_path=os.path.join(folder, "data", "techniquesEnterpriseAttack.json"),
            jq_schema=".[]",
            text_content=False,
        )

        docs = loader.load()
        print("\t- {} documentos cargados".format(len(docs)))

        print("Procesando documentos ...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        all_splits = text_splitter.split_documents(docs)
        print("\t- {} chunks creados".format(len(all_splits)))

        print("Indexando documentos ...")

        if embedder == "OpenAIEmbeddings":
            load_dotenv()  # carga OPENAI_API_KEY del fichero .env
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]
            )
        elif embedder == "HuggingFaceEmbeddings":
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"El embedder {embedder} no es válido")

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        vector_store = Chroma(
            collection_name="cve_collection",
            embedding_function=embeddings,
            persist_directory=destitationFolder,  # Where to save data locally, remove if not necessary
        )

        vector_store.add_documents(documents=all_splits)
        print(f"\t- indice {destitationFolder} creado")


def loadVectorStore(storeFolder: str = "./chroma_cve_db") -> Chroma:
    """
    Load the vector store.

    Args:
        - storeFolder (str): The folder where the vector store is saved.

    Returns:
        - Chroma: The vector store.
    """
    createVectorStore(destitationFolder=storeFolder)

    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return Chroma(
        collection_name="cve_collection",
        embedding_function=embeddings,
        persist_directory=storeFolder,
    )


if __name__ == "__main__":
    createVectorStore(createAgain=True)
