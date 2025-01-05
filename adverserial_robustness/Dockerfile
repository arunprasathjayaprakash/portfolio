FROM python:3.9.20-slim-bullseye

COPY . ./adverserial_robustness

WORKDIR /adverserial_robustness

RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0

ENTRYPOINT ["streamlit", "run", "scripts/code_base.py", "--server.port=8504","--server.address=0.0.0.0"]