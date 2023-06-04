import logging
import os
import runpod
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

repo_file = hf_hub_download(repo_id=os.environ["GGML_REPO"], filename=os.environ["GGML_FILE"], revision=os.environ.get("GGML_REVISION", "main"))

llm = AutoModelForCausalLM.from_pretrained(repo_file, model_type=os.environ.get("GGML_TYPE", "llama"), gpu_layers=int(os.environ.get("GGML_LAYERS", 0)))

def inference(event):
    job_input = event["input"]
    prompt = job_input.pop("prompt")
    return llm(prompt, **job_input)

runpod.serverless.start({"handler": inference})