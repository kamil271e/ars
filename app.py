import streamlit as st
from src.rag import *


@st.cache_resource(max_entries=1)
def load_rag_component() -> RAG:
    cfg = Config()
    return RAG(
        data_dir=cfg.DATA_DIR,
        dataset=cfg.DATASET,
        embeddings_model=cfg.EMBEDDINGS_MODEL,
        vectorstore_dir=cfg.VECTORSTORE_DIR,
        vectorstore_type=cfg.VECTORSTORE_TYPE,
        text_splitter=cfg.TEXT_SPLITTER,
    )


if __name__ == "__main__":
    img_col, title_col = st.columns([1, 2])
    with img_col:
        st.image(
            "https://findingtom.com/images/uploads/medium-logo/article-image-00.jpeg",
            width=200,
        )
    with title_col:
        st.markdown('<p style="font-size: 36px; margin-top: 30px;">ARS: Article Retrieval System</p>', unsafe_allow_html=True)

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

    rag = load_rag_component()

    if st.button("Retrieve"):
        results = rag.simple_retrieval(query, n=n)
        if results:
            expanders = [0] * len(results)
            st.subheader("Results:")
            for i, result in enumerate(results):
                expanders[i] = st.expander(result.page_content)
                expanders[i].write(result.metadata["Text"])
