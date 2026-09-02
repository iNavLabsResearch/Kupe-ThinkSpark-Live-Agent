# CUDA 12.4 + cuDNN runtime — matches the cu124 torch wheels below.
# Works on L4, H100, RTX 6000, and consumer 3060/4060/3090/4090/5090.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      libsndfile1 libportaudio2 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && python -m pip install --upgrade pip

WORKDIR /app

# torch pinned to the CUDA 12.4 wheels; >=2.5 because transformers>=4.56 requires it
RUN pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
VOLUME ["/cache/hf"]

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
