import streamlit as st
from rag import *


@st.cache_resource(max_entries=1)
def load_components() -> VectorStore:
    return load_vectorstore()


if __name__ == "__main__":
    # img_col, title_col = st.columns([1, 2])
    # with img_col:
    #     st.image(
    #         "https://findingtom.com/images/uploads/medium-logo/article-image-00.jpeg",
    #         width=200,
    #     )
    # with title_col:
    #     st.title("Article Retrieval System")
    st.title("Medium Article Retrieval System")

    query_col, k_col = st.columns([2, 1])
    with query_col:
        query = st.text_input("Enter your query:", "KNN introduction")
    with k_col:
        k = st.slider(
            "Select the number of results to obtain:",
            min_value=1,
            max_value=10,
            value=1,
        )

    vectorstore = load_components()

    if st.button("Retrieve"):
        results = simple_retrieval(vectorstore, query, k=k)
        if results:
            expanders = [0] * len(results)
            st.subheader("Results:")
            for i, result in enumerate(results):
                expanders[i] = st.expander(result.page_content)
                expanders[i].write(result.metadata["Text"])
