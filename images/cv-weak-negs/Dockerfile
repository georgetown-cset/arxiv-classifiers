FROM apache/beam_python3.9_sdk:2.46.0

RUN apt update && apt install -y \
    build-essential \
    curl \
    git \
    wget \
    vim

RUN apt install -y \
    apt-transport-https \
    gnupg

RUN pip install --upgrade pip wheel setuptools
RUN pip install --upgrade torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade transformers==4.24.0 sentencepiece==0.1.97 'protobuf<3.20'

# Transformers config
COPY config.json /inference/config.json

# Model assets
COPY model/model.pth /model/model.pth
COPY model/config.json /model/config.json

ENTRYPOINT ["/opt/apache/beam/boot"]
