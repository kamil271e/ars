# article-retrieval-system
Efficient RAG retrieval system for article fragments from Kaggle [dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset/data).


### Data loading
In order to download dataset, you can choose one of two options:
1. Download it manually from [link](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset/data) and create folder named ```data```. Then store the ```medium.csv``` file in that folder.
2. Use the Kaggle API: Download your account token from [this link](https://www.kaggle.com/settings/account), overwrite the existing ```kaggle.json``` file from repository and execute the following code, that sets up authentication:
```
mkdir -p ~/.kaggle/
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

