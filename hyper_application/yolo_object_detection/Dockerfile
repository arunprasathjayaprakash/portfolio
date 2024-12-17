FROM python:3

WORKDIR  /scripts

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY /scripts ./scripts
EXPOSE  8080

ENTRYPOINT ["streamlit", "run", "scripts/code_base.py", "--server.port=8080","--server.address=0.0.0.0"]