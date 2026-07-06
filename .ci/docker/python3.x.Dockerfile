ARG Python_VERSION=python:3.13-slim-trixie

FROM ${Python_VERSION}

COPY requirements.txt ./
COPY docs/requirements.txt ./docs/

RUN pip install -r requirements.txt
