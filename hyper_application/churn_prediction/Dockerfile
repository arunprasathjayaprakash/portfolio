FROM python:3.9.20-slim-bullseye

COPY . ./churn_prediction

WORKDIR /churn_prediction

RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0

ENTRYPOINT ["streamlit", "run", "scripts/app.py", "--server.port=8503","--server.address=0.0.0.0"]