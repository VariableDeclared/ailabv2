# vLLM quick setup

## Follow vLLM install with uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
# CUDA Based lab
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

## vLLM backend (Plain vLLM server)

Grab the template file
```
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/tool_chat_template_llama3.1_json.jinja
```

Run with llama:

```
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template tool_chat_template_llama3.1_json.jinja \
    --max-model-len=40608
```
