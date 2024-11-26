FROM python:3.9-slim

RUN #apk add --no-cache build-base gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR  ./

COPY requirements.txt ./

RUN pip install pyarrow==12.0.1 --no-cache-dir

RUN pip install -r requirements.txt

COPY ./deploy_cloud_run.py deploy_cloud_run.py
EXPOSE  8080

ENTRYPOINT ["python","deploy_cloud_run.py"]