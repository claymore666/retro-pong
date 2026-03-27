FROM python:3.12-slim

RUN pip install --no-cache-dir numpy && \
    apt-get update && apt-get install -y --no-install-recommends alsa-utils && \
    rm -rf /var/lib/apt/lists/*

COPY pong.py /app/pong.py
WORKDIR /app

ENV TERM=xterm-256color
ENV ESCDELAY=25

CMD ["python3", "pong.py"]
