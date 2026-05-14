import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_data")


def create_vector_database():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Vector database created successfully")
    print("Total documents:", len(documents))
    print("Total chunks:", len(chunks))


if __name__ == "__main__":
    create_vector_database()