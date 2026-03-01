# Compiling for CUDA 13.1

## Create virtual env
```
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```
## Clone and build
```
git clone https://github.com/vllm-project/vllm.git
cd vllm
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu131
uv pip install --editable .
```
