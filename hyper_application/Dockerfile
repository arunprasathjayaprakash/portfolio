FROM python:3.9.20-slim

RUN mkdir -p /hyper

COPY app.py ./hyper
COPY requirements.txt ./hyper

WORKDIR /hyper

RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0


ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8505","--server.address=0.0.0.0"]