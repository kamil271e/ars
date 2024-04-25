FROM python:3.9-slim

WORKDIR /home

COPY requirements.txt /home/
RUN pip3 install -r requirements.txt

# Kaggle auth for data load
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

COPY .env /home/.env

COPY data /home/data
COPY src /home/src
COPY app.py /home/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

