import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import DataFrameLoader, CSVLoader
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import Chroma, Qdrant, VectorStore
from langchain_core.documents.base import Document
from typing import Union, List

try:
    from .utils import *
except ImportError:
    from utils import *

DATA_DIR = "data/medium.csv"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50


def data_loader(data_dir: str = DATA_DIR) -> Union[CSVLoader, DataFrameLoader]:
    df = pd.read_csv(data_dir)
    return DataFrameLoader(df, page_content_column="Title")
    # return CSVLoader(data_dir, source_column="Title") # much slower than DataFrameLoader


@tictoc
def load_vectorstore(
    vectorstore_path=VECTORSTORE_PATH, data_dir=DATA_DIR
):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    if os.path.exists(vectorstore_path):
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    else:
        loader = data_loader(data_dir)
        documents = loader.load()
        text_splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=vectorstore_path,
        )
    return vectorstore


def simple_retrieval(
    vectorstore: VectorStore, query: str, k: int = 2
) -> List[Document]:
    return vectorstore.similarity_search(query, k)


if __name__ == "__main__":
    retriever = load_vectorstore()
    query = "KNN Introduction"
    results = simple_retrieval(retriever, query)
    for result in results:
        print(result.page_content)
