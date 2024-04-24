import os
import pandas as pd
from langchain.text_splitter import TextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant, FAISS, VectorStore
from qdrant_client import QdrantClient
from typing import List
from src.utils import timer


class Config:
    DATA_PATH = "data/medium.csv"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 50
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    TEXT_SPLITTER = SemanticChunker(EMBEDDINGS_MODEL)
    VECTORSTORE_TYPE = Qdrant
    VECTORSTORE_PATH = "vectorstore4"


class RAG:
    def __init__(
        self,
        data_path: str,
        text_splitter: TextSplitter,
        embeddings_model: HuggingFaceEmbeddings,
        vectorstore_type: VectorStore,
        vectorstore_path: str,
    ):
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        self.text_splitter = text_splitter
        self.vectorstore_type = vectorstore_type
        self.embeddings_model = embeddings_model
        self.vectorstore = None
        self.load_vectorstore()

    @staticmethod
    def data_loader(data_path: str) -> DataFrameLoader:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # TODO: download from kaggle
            pass
        df = pd.read_csv(data_path)
        return DataFrameLoader(df, page_content_column="Title")

    @timer
    def load_vectorstore(self):
        if os.path.exists(self.vectorstore_path):
            self.init_vectorstore()
        else:
            self.create_vectorstore()

    def simple_retrieval(self, query: str, n: int) -> List[Document]:
        return self.vectorstore.similarity_search(query, n)

    def init_vectorstore(self):
        """
        Vectorstore loading method.
        Unfortunately LangChain API provide separate methods for loading different types of vectorstores.
        """
        if self.vectorstore_type == Chroma:
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings_model,
            )
        elif self.vectorstore_type == Qdrant:
            self.vectorstore = Qdrant(
                client=QdrantClient(path=self.vectorstore_path),
                embeddings=self.embeddings_model,
                collection_name="qdrant_collection",
            )
        elif self.vectorstore_type == FAISS:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        else:
            raise ValueError("Vectorstore type not supported.")

    def create_vectorstore(self):
        # INGEST
        loader = self.data_loader(self.data_path)
        documents = loader.load()
        documents = self.text_splitter.split_documents(documents)

        if self.vectorstore_type == Chroma:
            self.vectorstore = Chroma.from_documents(
                documents,
                self.embeddings_model,
                persist_directory=self.vectorstore_path,
            )
        elif self.vectorstore_type == Qdrant:
            self.vectorstore = Qdrant.from_documents(
                documents,
                self.embeddings_model,
                path=self.vectorstore_path,
                collection_name="qdrant_collection",
            )
        elif self.vectorstore_type == FAISS:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
            self.vectorstore.save_local(self.vectorstore_path)
        else:
            raise ValueError("Vectorstore type not supported.")


if __name__ == "__main__":
    config = Config()
    rag = RAG(
        data_path=config.DATA_PATH,
        embeddings_model=config.EMBEDDINGS_MODEL,
        vectorstore_path=config.VECTORSTORE_PATH,
        vectorstore_type=config.VECTORSTORE_TYPE,
        text_splitter=config.TEXT_SPLITTER,
    )
    results = rag.simple_retrieval("KNN introduction", n=10)
    for result in results:
        print(result.page_content)
