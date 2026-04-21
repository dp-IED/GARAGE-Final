FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RUN python -m pip install torch-geometric==2.6.1

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

COPY . /workspace

RUN chmod +x /workspace/docker/garage.sh

ENV PYTHONPATH=/workspace

ENTRYPOINT ["/workspace/docker/garage.sh"]
CMD ["shell"]
