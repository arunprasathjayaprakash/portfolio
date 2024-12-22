FROM python:3.9.20-slim-bullseye

COPY . ./Yolo_object_detection

COPY requirements.txt ./Yolo_object_detection

WORKDIR /Yolo_object_detection

RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8505","--server.address=0.0.0.0"]