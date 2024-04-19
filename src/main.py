import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import DataFrameLoader, CSVLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from tqdm import tqdm
from typing import Union, List
from langchain_core.documents.base import Document
from utils import *

DATA_DIR = "data/medium.csv"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50


def data_loader(data_dir: str = DATA_DIR) -> Union[CSVLoader, DataFrameLoader]:
    return CSVLoader(data_dir)


@tictoc
def fill_vectorstore(loader: Union[CSVLoader, DataFrameLoader]) -> None:
    documents = loader.load()
    text_splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n"
    )
    texts = text_splitter.split_text(documents)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    db = Chroma.from_documents(texts, embeddings, persist_directory=VECTORSTORE_PATH)
    db.persist()


def load_vectorstore() -> Union[Chroma, Qdrant]:
    if not os.path.exists(VECTORSTORE_PATH):
        fill_vectorstore(CSVLoader(DATA_DIR))
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    return Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)


def simple_retrieval(vectorstore: Union[Chroma, Qdrant], query: str, k: int=2) -> List[Document]:
    return vectorstore.similarity_search(query, k)


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    query = "KNN Introduction"
    results = simple_retrieval(vectorstore, query)
    for result in results:
        print(result.page_content)