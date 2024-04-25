# ARS: Report

In the following report, I outline the tools, strategies, and motivations employed during the development of this project. The system supports language models compatible with HuggingFace, along with vector stores supported by the LangChain framework. To explore different options, feel free to experiment with the ```Config``` class in ```src/rag.py```.

## Framework
I opted for LangChain as my preferred library for RAG due to its robust documentation, adaptability, and my familiarity with it. It is also scalable and user-friendly. While LLamaIndex was also considered, I ultimately chose LangChain because I believe it is more flexible and can integrate more things easily.

## Vector store
Considering their widespread popularity, I explored and integrated support for three vector stores: **ChromaDB**, **FAISS** and **QDrant**. While they share similarities, each demonstrates some distinct characteristics.

Default distance functions used for similarity measure are as follows - for Chroma and FAISS: L2  (Euclidian), and for QDrant: cosine similarity. Although the document ingestion times for Chroma and FAISS were similar, QDrant took significantly longer. However, the retrieval times for each were comparable.

Indexing strategy is a crucial process in the selection of optimal vector store. By default, both Chroma and QDrant utilize a graph-based index (HNSW) that does not guarantee 100% recall or accuracy of retrieval. In contrast, FAISS employs FLAT, a brute-force index that meets this objective. Given that the examined dataset is relatively small (~8MB) which suggests that the ingesting process as well as retrieval should not be computationally expensive, I concluded that FAISS would be the most suitable choice in this scenario.

## Chunking strategy
While initially, I experimented with basic recursive and token splitters offered in LangChain, I believe that using context-aware chunking is generally a more beneficial approach than trying to fine-tune/tweak parameters that regard chunk size and overlap. I am of the opinion that it should be appropriate when we care about the content richness of chunks. To incorporate semantic understanding of the text into this process I utilize ``SemanticChunker`` module. It requires an additional embedding model to calculate, based on some similarity measure, whether a part of the text should be included in a given chunk or if its meaning differs slightly.

## Embeddings model
For the purpose of generating embeddings for vector store, I have chosen [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from the sentence transformers family. It is very popular and achieves decent retrieval results on [MTEB](https://huggingface.co/spaces/mteb/leaderboard) benchmark. Most importantly it achieves sufficient results with a small number of parameters: 26M. 

## LLM
I will briefly discuss my choices for a Q/A system that integrates with vector store in RAG fashion.

Mistral7B is known for its remarkable results with a small size, vastly due to the sliding window attention mechanism, rolling buffer cache, and pre-fill with chunking. I have decided to use its refined version: [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) which employs a very engaging concept of a mixture of experts. Instead of using only one linear layer after propagation through attention heads, it leverages a router network that for each input token chooses a few [by default 2] linear networks and aggregates results from it. This process allows to use of just a subset of all weights during inference. For more information about this compelling model, I strongly encourage you to get familiar with [repository](https://github.com/mistralai/mistral-src) and [paper](https://arxiv.org/pdf/2401.04088).

I have also tested the recently announced [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), but retrieval performance was not influenced much, therefore I have decided to stick with Mixtral.

I could have used the model locally and employed quantization techniques for more effective inference, instead, I have decided that the time of response of my system is crucial, and with computational resources constraints on my local machine, it will be the most convenient to leverage HuggingFace inference API for the arbitrarily chosen model. I consider this choice the simplest and fastest way to benchmark the performance of large models.


## Future perspectives
One of the possible further enhancements might regard exploring more advanced chunking techniques involving external agents, such as agentic chunking. Additionally, exploring an approach to index chunks based on their summaries could prove beneficial, especially given the presence of redundant information or metadata in the examined articles. Lastly, experimenting with more refined embedding models and LLMs, such as those from OpenAI, is also worth examining.

## Resources
* [LangChain API](https://api.python.langchain.com/en/latest/langchain_api_reference.html)
* [ Practical RAG - Choosing the Right Embedding Model, Chunking Strategy, and More ](https://www.youtube.com/watch?v=j1XRLh7yzzY)
* [ Chunking Strategies in RAG: Optimising Data for Advanced AI Responses ](https://www.youtube.com/watch?v=pIGRwMjhMaQ)

