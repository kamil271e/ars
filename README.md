# article-retrieval-system
Efficient RAG retrieval system for article fragments from the Kaggle dataset [available here](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset/data). This system supports popular vector stores retrieval and includes a Question Answering functionality with Large Language Models (LLMs). By default, it utilizes [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) for vector store retrieval and the [Mixtral-8x7B](https://arxiv.org/pdf/2401.04088) LLM. More details in [**report**](report.md).

## Prerequisites

### Data loading
To download the dataset, you can choose one of two options:
1. Download it manually from [link](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset/data) and create folder named ```data``` in project root directory. Then store the ```medium.csv``` file in that folder.
2. Use the Kaggle API: Download your account token from [this link](https://www.kaggle.com/settings/account) and overwrite the existing ```kaggle.json``` file.


### User Access Token
This step is not obligatory but necessary if you want to use Q/A system with Large Language Model support. To obtain your HuggingFaceHub API Token generate it and copy it from your HuggingFace [account](https://huggingface.co/settings/tokens) and paste it to ``.env`` file overwriting ``<YOUR_TOKEN>`` placeholder.

## Running options

### Local
#### Poetry setup
```
pip install pipx
pipx install poetry
poetry install
poetry shell
```
#### Running
```
chmod +x kaggle.sh
./kaggle.sh
streamlit run app.py
```

### Docker
```
sudo docker build -t ars-app:latest .
sudo docker container run -it -p 8501:8501 ars-app:latest
```

## Demo
![ars_demo](https://github.com/kamil271e/ars/assets/82380348/a34596ca-e5a4-48fc-bfcf-fe4729b78fbc)
