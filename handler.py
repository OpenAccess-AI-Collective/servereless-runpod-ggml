import logging
import os
from typing import Generator

import runpod
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

repo_file = hf_hub_download(repo_id=os.environ["GGML_REPO"], filename=os.environ["GGML_FILE"], revision=os.environ.get("GGML_REVISION", "main"))
llm = None


def get_llm():
    global llm
    if not llm:
        llm = AutoModelForCausalLM.from_pretrained(repo_file, model_type=os.environ.get("GGML_TYPE", "llama"), gpu_layers=int(os.environ.get("GGML_LAYERS", 0)))
    return llm


def inference(event):
    job_input = event["input"]
    stream = event["stream"] if "stream" in event else True
    prompt: str = job_input.pop("prompt")
    llm_res: Generator[str, None, None] = get_llm()(prompt, stream=stream, **job_input)
    if stream:
        for res in llm_res:
            yield res
    else:
        return llm_res


runpod.serverless.start({"handler": inference})