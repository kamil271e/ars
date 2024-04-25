import streamlit as st

from typing import Tuple
from src.rag import *


@st.cache_resource(max_entries=1)
def rag_components() -> Tuple[RAG, Config]:
    cfg = Config()
    return (
        RAG(
            data_dir=cfg.DATA_DIR,
            dataset=cfg.DATASET,
            embeddings_model=cfg.EMBEDDINGS_MODEL,
            vectorstore_dir=cfg.VECTORSTORE_DIR,
            vectorstore_type=cfg.VECTORSTORE_TYPE,
            text_splitter=cfg.TEXT_SPLITTER,
        ),
        cfg,
    )


def vector_store_retrieval_components() -> Tuple[str, int]:
    img_col, title_col = st.columns([1, 2])
    with img_col:
        st.image(
            "https://findingtom.com/images/uploads/medium-logo/article-image-00.jpeg",
            width=200,
        )
    with title_col:
        st.markdown(
            '<p style="font-size: 36px; margin-top: 30px;">ARS: Article Retrieval System</p>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="font-size: 36px; margin-top: 30px;">Vector Store Retrieval</p>',
        unsafe_allow_html=True,
    )
    query_col, n_col = st.columns([2, 1])
    with query_col:
        query = st.text_input("Enter your query:", "KNN introduction")
    with n_col:
        n = st.slider(
            "Select the no. of results to retrieve:",
            min_value=1,
            max_value=10,
            value=1,
        )
    return query, n


def qa_system_components() -> Tuple[str, int, int]:
    st.markdown(
        '<p style="font-size: 36px; margin-top: 30px;">Q/A System</p>',
        unsafe_allow_html=True,
    )
    question = st.text_input("Enter your question:", "What is Variance?")
    max_tokens = st.slider(
        "Select the max no. of tokens:",
        min_value=50,
        max_value=150,
        value=90,
    )
    chunks = st.slider(
        "Select the no. of used chunks:",
        min_value=1,
        max_value=5,
        value=2,
    )
    return question, max_tokens, chunks


if __name__ == "__main__":
    query, n = vector_store_retrieval_components()
    rag, cfg = rag_components()

    if st.button("Retrieve"):
        results = rag.simple_retrieval(query, n=n)
        if results:
            expanders = [0] * len(results)
            st.subheader("Results:")
            for i, result in enumerate(results):
                expanders[i] = st.expander(result.page_content)
                expanders[i].write(result.metadata["Text"])

    question, max_tokens, chunks = qa_system_components()

    if st.button("Ask LLM"):
        if not cfg.HUGGINGFACEHUB_API_TOKEN:
            st.error(
                "Please set the HUGGINGFACEHUB_API_TOKEN environment variable in .env file."
            )
            st.stop()
        else:
            results = rag.generate_llm_answer(
                api=cfg.LLM_API,
                token=cfg.HUGGINGFACEHUB_API_TOKEN,
                question=question,
                num_chunks=chunks,
                max_tokens=max_tokens,
            )
            if results:
                st.write(results)
