FROM python:3.12-slim-bookworm

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
make

COPY requirements.txt ./

RUN pip install -r requirements.txt
