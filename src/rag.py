import os
import requests
import pandas as pd
import openai

from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant, FAISS, VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents.base import Document
from qdrant_client import QdrantClient

from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
from typing import List
from src.utils import timer


class Config:
    load_dotenv()
    DATA_DIR = "data"
    DATASET = "meruvulikith/1300-towards-datascience-medium-articles-dataset"
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    TEXT_SPLITTER = SemanticChunker(EMBEDDINGS_MODEL)
    VECTORSTORE_TYPE = FAISS
    VECTORSTORE_DIR = "vectorstore"
    LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    LITELLM_URL = "http://0.0.0.0:4000"
    LITELLM_MODEL = "llama3"
    OPENAI_MODEL = "gpt-3.5-turbo-0125"


class RAG:
    def __init__(
        self,
        data_dir: str,
        dataset: str,
        text_splitter: TextSplitter,
        embeddings_model: HuggingFaceEmbeddings,
        vectorstore_type: VectorStore,
        vectorstore_dir: str,
    ):
        self.data_dir = data_dir
        self.dataset = dataset
        self.vectorstore_dir = vectorstore_dir
        self.text_splitter = text_splitter
        self.vectorstore_type = vectorstore_type
        self.embeddings_model = embeddings_model
        self.vectorstore = None
        self.load_vectorstore()

    def get_data_loader(self) -> DataFrameLoader:
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            dataset = self.dataset
            kaggle_api.dataset_download_files(dataset, path=self.data_dir, unzip=True)
        files = os.listdir(self.data_dir)
        file_path = os.path.join(self.data_dir, files[0])
        df = pd.read_csv(file_path)
        return DataFrameLoader(df, page_content_column="Title")

    @timer
    def load_vectorstore(self):
        if os.path.exists(self.vectorstore_dir):
            self.init_vectorstore()
        else:
            self.create_vectorstore()

    def simple_retrieval(self, query: str, n: int) -> List[Document]:
        return self.vectorstore.similarity_search(query, n)

    def init_vectorstore(self):
        """
        Vectorstore loading method.
        This method initializes the vector store based on the specified vector store type.
        Unfortunately, LangChain API does not provide a universal method for loading/creating different vector stores.
        Supports: Chroma, Qdrant, FAISS
        """
        if self.vectorstore_type == Chroma:
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embeddings_model,
            )
        elif self.vectorstore_type == Qdrant:
            self.vectorstore = Qdrant(
                client=QdrantClient(path=self.vectorstore_dir),
                embeddings=self.embeddings_model,
                collection_name="qdrant_collection",
            )
        elif self.vectorstore_type == FAISS:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_dir,
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        else:
            raise ValueError("Vectorstore type not supported.")

    def create_vectorstore(self):
        """
        Vectorstore creation method.
        This method loads the data using a data loader, splits the documents using a text splitter,
        and then ingests the document embeddings into the chosen vector store.
        """
        loader = self.get_data_loader()
        documents = loader.load()
        documents = self.text_splitter.split_documents(documents)

        if self.vectorstore_type == Chroma:
            self.vectorstore = Chroma.from_documents(
                documents,
                self.embeddings_model,
                persist_directory=self.vectorstore_dir,
            )
        elif self.vectorstore_type == Qdrant:
            self.vectorstore = Qdrant.from_documents(
                documents,
                self.embeddings_model,
                path=self.vectorstore_dir,
                collection_name="qdrant_collection",
            )
        elif self.vectorstore_type == FAISS:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
            self.vectorstore.save_local(self.vectorstore_dir)
        else:
            raise ValueError("Vectorstore type not supported.")

    def generate_llm_answer(
        self, api: str, token: str, question: str, num_chunks: int, max_tokens: int
    ) -> str:
        if token is None:
            raise ValueError("HuggingFace API token is required.")
        headers = {"Authorization": f"Bearer {token}"}

        def query(payload):
            response = requests.post(api, headers=headers, json=payload)
            return response.json()

        retrieved = self.simple_retrieval(question, num_chunks)
        context = " ".join([doc.metadata["Text"] for doc in retrieved])

        prompt = f"""
        [INST] 
        Answer the question, use your knowledge and given context:

        {context}

        QUESTION:
        {question} 

        [/INST]"""

        output = query({"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}})
        generated_text = output[0]["generated_text"]
        end_flag = "[/INST]"
        inst_index = generated_text.find(end_flag)
        return generated_text[inst_index + len(end_flag) :].strip()

    def generate_llm_answer_openai(
        self, api: str, model: str, question: str, num_chunks: int, max_tokens: int
    ) -> str:
        client = openai.Client(api_key="123", base_url=api)

        retrieved = self.simple_retrieval(question, num_chunks)
        context = " ".join([doc.metadata["Text"] for doc in retrieved])

        prompt = f"""
        [INST] 
        Answer the question, use your knowledge and given context:

        {context}

        QUESTION:
        {question} 

        [/INST]"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
