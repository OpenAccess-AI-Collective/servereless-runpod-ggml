#!/bin/bash

python3 -c "from huggingface_hub import hf_hub_download;hf_hub_download(repo_id='${GGML_REPO}', filename='${GGML_FILE}', revision='${GGML_REVISION}')"

# Execute the passed arguments (CMD)
exec "$@"
