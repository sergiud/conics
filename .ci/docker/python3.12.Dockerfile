FROM python:3.12-slim-bookworm

COPY requirements.txt ./
COPY docs/requirements.txt ./docs/

RUN pip install -r requirements.txt
