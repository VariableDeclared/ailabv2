# vLLM Model Server

A Python FastAPI server for hosting Hugging Face models using vLLM for efficient inference.

## Features

- Load and host Hugging Face models efficiently using vLLM
- RESTful API for model management and text generation
- Support for multiple concurrent models
- Streaming generation options
- Automatic model loading and unloading
- CORS support for web clients

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with default settings:

```bash
python vllm_server.py
```

Run with a specific model:

```bash
python vllm_server.py --model meta-llama/Llama-2-7b-chat-hf
```

Run with custom host and port:

```bash
python vllm_server.py --model facebook/bart-large-cnn --port 8001 --host 0.0.0.0
```

## API Endpoints

### Health Check
- `GET /` - Server information and available endpoints

### Model Management
- `GET /models/available` - List all available models from Hugging Face Hub
- `GET /models` - List all currently loaded models
- `POST /configure` - Configure model parameters
- `GET /models/{model_name}` - Load a specific model
- `POST /models/{model_name}/unload` - Unload a loaded model
- `POST /models/batch_load` - Load multiple models in parallel

### Generation
- `POST /generate` - Generate text using a loaded model

## Examples

### Load a Model

```bash
curl -X GET "http://localhost:8000/models/meta-llama/Llama-2-7b-chat-hf"
```

### List Loaded Models

```bash
curl -X GET "http://localhost:8000/models"
```

### Generate Text

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "Once upon a time",
         "temperature": 0.7,
         "max_tokens": 100
     }'
```

### List Available Models

```bash
curl -X GET "http://localhost:8000/models/available"
```

## Configuration

You can configure models via POST requests to `/configure`:

```bash
curl -X POST "http://localhost:8000/configure" \
     -H "Content-Type: application/json" \
     -d '{
         "model_name": "my-custom-model",
         "gpu_memory_utilization": 0.8,
         "max_model_len": 2048
     }'
```

## Features

- **Efficient Inference**: Uses vLLM's PagedAttention architecture for optimal throughput
- **Async API**: Non-blocking operations for better performance
- **Model Pooling**: Load multiple models for different tasks
- **Flexible Configuration**: Customize model parameters dynamically

## Requirements

- Python 3.8+
- PyTorch
- vLLM
- FastAPI
- Uvicorn
- Hugging Face Hub library

## Notes

- GPU memory requirement depends on the model size and configuration
- Some models require you to have access permissions on Hugging Face Hub
- Models are loaded in-memory and remain loaded until explicitly unloaded

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.