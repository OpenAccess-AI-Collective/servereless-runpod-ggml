ARG CUDA_VERSION="11.8.0"
ARG UBUNTU_VERSION="22.04"
FROM winglian/runpod-serverless-ggml-base:latest as ctransformers-base

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu$UBUNTU_VERSION

ARG PYTHON_VERSION="3.9"
ENV PYTHON_VERSION=$PYTHON_VERSION

RUN useradd -m -u 1001 appuser
RUN apt-get update && \
    apt-get install -y wget git && rm -rf /var/lib/apt/lists/* && \
    mkdir /runpod-volume && \
    chown appuser:appuser /runpod-volume && \
    mkdir /workspace && \
    chown appuser:appuser /workspace && \
    mkdir /app && \
    chown appuser:appuser /app

USER appuser

ENV HOME /home/appuser
ENV PATH="${HOME}/miniconda3/bin:${PATH}"
WORKDIR /app
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ${HOME}/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="${HOME}/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}:/app"


RUN mkdir -p /app/builds
COPY --from=ctransformers-base /workspace/builds/ctransformers /app/builds/ctransformers

ADD requirements.txt .
RUN pip3 install -r requirements.txt && \
    pip3 install /app/builds/ctransformers/dist/*.whl && \
    rm requirements.txt

# Add your file
ADD handler.py .
ADD entrypoint.sh .

ENV GGML_REPO=""
ENV GGML_FILE=""
ENV GGML_TYPE="llama"
ENV GGML_LAYERS="32"
ENV GGML_REVISION="main"

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

ENTRYPOINT [ "entrypoint.sh" ]

# Call your file when your container starts
CMD [ "python3", "-u", "handler.py" ]
