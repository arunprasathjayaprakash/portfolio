FROM python:3.9.20-slim-bullseye

COPY . ./contrastive_learning

WORKDIR  /contrastive_learning

RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0


ENTRYPOINT ["streamlit", "run", "scripts/app.py", "--server.port=8501","--server.address=0.0.0.0"]