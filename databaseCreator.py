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
    destitationFolder: str = "./chroma_mitre_db",
    createAgain: bool = False,
    embedder: str = "HuggingFaceEmbeddings",
) -> None:
    """
    Create a vector store with the mitres.

    Args:
        - destitationFolder (str): The folder where the vector store will be saved.
        - createAgain (bool): If True, the vector store will be created again,
            even if it already exists.
        - embedder (str): The embedder to use. It can be "HuggingFaceEmbeddings" or "OpenAIEmbeddings".

    Returns:
        - None
    """

    def extract_metadata(record: dict, metadata: dict) -> dict:
        """
        This is where all the information that we do not index by needs
        to be stored so the model can use it later.

        Args:
            - record (dict): The record.
            - metadata (dict): The metadata.

        Returns:
            - dict: The metadata.
        """
        # It is exists we remove the source because we only have one json file
        if "source" in metadata:
            del metadata["source"]

        # We also remove the seq_num because it is not useful
        if "seq_num" in metadata:
            del metadata["seq_num"]

        # This is the id of the MITRE technique
        metadata["id"] = record["id"]

        # This is the URL of the MITRE technique
        metadata["url"] = record["url"]

        # This is the name of the MITRE technique
        metadata["name"] = record["name"]

        # This are the tactics used by the MITRE technique
        tactics = record.get("tactics", [])
        metadata["tactics"] = ", ".join(tactics)

        # This are the platforms affected by the MITRE technique
        platforms = record.get("platforms", [])
        metadata["platforms"] = ", ".join(platforms)

        return metadata

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
            content_key="description",  # It is only indexing by the description
            # It only searches by a fragment and only stores that fragment
            metadata_func=extract_metadata,
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
            collection_name="mitre_collection",
            embedding_function=embeddings,
            persist_directory=destitationFolder,  # Where to save data locally, remove if not necessary
        )

        vector_store.add_documents(documents=all_splits)
        print(f"\t- indice {destitationFolder} creado")


def loadVectorStore(storeFolder: str = "./chroma_mitre_db") -> Chroma:
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
        collection_name="mitre_collection",
        embedding_function=embeddings,
        persist_directory=storeFolder,
    )


if __name__ == "__main__":
    createVectorStore(createAgain=True)
