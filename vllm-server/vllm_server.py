from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
# from vllm.serving.async_generation import AsyncEngineArgs as VLLMEngineArgs
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    model_name: str
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9


class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int = 256
    stop: List[str] = []
    stream: bool = False
    echo: bool = False
    model_name: str


class GenerationResponse(BaseModel):
    model: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    finish_reason: str


class ServerState:
    def __init__(self):
        self.loaded_models: Dict[str, AsyncLLMEngine] = {}
        self.is_model_loading: Dict[str, asyncio.Event] = {}
        self.model_configs: Dict[str, ModelConfig] = {}

# Create a global variable or environment check for the initial model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "facebook/opt-125m")
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_models = {}
    app.state.is_model_loading = {}
    app.state.model_configs = {}
    # Initialize the default config HERE
    logger.info(f"Pre-configuring default model: {DEFAULT_MODEL}")
    initial_config = ModelConfig(model_name=DEFAULT_MODEL)
    app.state.model_configs[DEFAULT_MODEL] = initial_config
    logger.info("Server starting...")
    yield
    logger.info("Server shutting down...")


app = FastAPI(
    title="vLLM Model Server",
    description="Server for hosting Hugging Face models using vLLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "vLLM Model Server",
        "version": "1.0.0",
        "endpoints": {
            "/models": "List all loaded models",
            "/models/available": "Get list of all available models from HF Hub",
            "/model/{model_name}": "Load a specific model",
            "/generate": "Generate text with a loaded model",
            "/model/{model_name}/unload": "Unload a model",
        },
    }


@app.get("/models/available")
async def list_available_models():
    """List all available models from Hugging Face Hub"""
    from huggingface_hub import HfApi
    api = HfApi()
    

    try:
        models = api.list_models()
        return {
            "available_models": [model.id for model in models],
            "count": sum(1 for _ in models),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models")
async def list_loaded_models():
    """List all currently loaded models"""
    if not app.state.loaded_models:
        return {"loaded_models": [], "count": 0}

    models_info = []
    for model_name, engine in app.state.loaded_models.items():
        config = app.state.model_configs.get(model_name)
        state = "loaded" if engine else "unloaded"
        models_info.append(
            {
                "model_name": model_name,
                "state": state,
                "config": config.dict() if config else None,
            }
        )

    return {"loaded_models": models_info, "count": len(app.state.loaded_models)}


@app.get("/model/{model_repo}/{model_name}")
async def load_model(model_repo, model_name: str):
    # import pdb; pdb.set_trace()
    """Load a model from Hugging Face Hub"""
    model_name = f"{model_repo}/{model_name}"
    if model_name in app.state.loaded_models:
        return {
            "message": f"Model {model_name} is already loaded",
            "model_name": model_name,
        }

    if model_name in app.state.is_model_loading:
        event = app.state.is_model_loading[model_name]
        try:
            await event.wait()
        except Exception as e:
            return {"error": f"Failed to load model {model_name}: {str(e)}"}

    try:
        event = asyncio.Event()
        app.state.is_model_loading[model_name] = event

        config = app.state.model_configs.get(model_name)

        async def load():
            try:
                engine_args = AsyncEngineArgs(
                    model=model_name,
                    tokenizer_mode=config.tokenizer_mode if config else "auto",
                    trust_remote_code=config.trust_remote_code if config else False,
                    dtype=config.dtype if config else "auto",
                    max_model_len=config.max_model_len if config else None,
                    gpu_memory_utilization=config.gpu_memory_utilization
                    if config
                    else 0.9,
                )
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                app.state.loaded_models[model_name] = engine
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
            finally:
                app.state.is_model_loading[model_name] = event
                event.set()

        asyncio.create_task(load())

        return {
            "message": f"Loading model {model_name}...",
            "model_name": model_name,
            "status": "loading",
        }
    except Exception as e:
        return {"error": f"Failed to initiate model loading: {str(e)}"}


@app.post("/models/{model_repo}/{model_name}/unload")
async def unload_model(model_repo, model_name: str):
    """Unload a loaded model"""
    model_name = f"{model_repo}/{model_name}"
    if model_name not in app.state.loaded_models:
        return {"error": f"Model {model_name} is not loaded", "model_name": model_name}

    engine = app.state.loaded_models[model_name]

    try:
        await engine.stop_background_loop()
        app.state.loaded_models[model_name] = None
        if model_name in app.state.model_configs:
            del app.state.model_configs[model_name]
        logger.info(f"Successfully unloaded model: {model_name}")
        return {
            "message": f"Model {model_name} unloaded successfully",
            "model_name": model_name,
        }
    except Exception as e:
        return {"error": f"Failed to unload model {model_name}: {str(e)}"}


@app.post("/configure", response_model=ModelConfig)
async def configure_model(config: ModelConfig):
    """Configure model parameters"""
    app.state.model_configs[config.model_name] = config
    return config


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text using a loaded model"""
    request_id = "0" # request.headers.get("X-Request-ID", str(uuid.uuid4()))
    engine = app.state.loaded_models.get(request.model_name)
    if not engine or engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_name} not loaded. Please load it first.",
        )

    try:
        
        sampling_params = {
            "temperature": float(request.temperature),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens,
            # "stop": request.stop,
        }

        results_generator = engine.generate(
            request.prompt,
            SamplingParams(**sampling_params),
            request_id,
        )
        result = None
        async for request_output in results_generator:
            # TODO Client disconnect logic
            # if await request.is_disconnected():
            #     # Abort the request if the client disconnects.
            #     await engine.abort(request_id)
            result = request_output
        output_text = result.outputs[0].text if result.outputs else ""
        # import pdb; pdb.set_trace()
        return GenerationResponse(
            model=request.model_name,
            generated_text=output_text,
            prompt_tokens=len(result.prompt_token_ids),
            generated_tokens=result.metrics.num_generation_tokens,
            total_tokens=len(result.prompt_token_ids) + result.metrics.num_generation_tokens,
            finish_reason=str(result.outputs[0].finish_reason) if result.outputs else "stop",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/models/batch_load", response_model=Dict[str, Dict[str, str]])
async def batch_load_models(models: List[str]):
    """Load multiple models in parallel"""
    results = {}
    for model_name in models:
        result = await load_model(model_name)
        results[model_name] = result
        if "error" in result:
            break

    return {
        "results": results,
        "total_requested": len(models),
        "successful": sum(1 for r in results.values() if "error" not in r),
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, model_name: str = "default"):
    """Run the vLLM server"""
    import argparse

    def run_async():
        uvicorn.run(
            "vllm_server:app", host=host, port=port, reload=False, log_level="info"
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_async())
    finally:
        loop.run_until_complete(loop.shutdown_all_tasks())
        loop.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("vLLM Model Server")
            print("\nUsage:")
            print(
                "  python vllm_server.py [--model MODEL_NAME] [--port PORT] [--host HOST]"
            )
            print("\nExamples:")
            print("  python vllm_server.py --model meta-llama/Llama-2-7b-chat-hf")
            print("  python vllm_server.py --model facebook/bart-large-cnn --port 8001")
            sys.exit(0)

        import argparse

        parser = argparse.ArgumentParser(description="vLLM Model Server")
        parser.add_argument(
            "--model",
            type=str,
            default="default",
            help="Model to load from Hugging Face",
        )
        parser.add_argument(
            "--port", type=int, default=8000, help="Port to run the server on"
        )
        parser.add_argument(
            "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
        )
        args = parser.parse_args()

        run_server(args.host, args.port, args.model)
    else:
        print("Starting vLLM Model Server...")
        print(
            "To load a specific model, run: python vllm_server.py --model <model_name>"
        )
        print("Server will load 'default' model")
        run_server("0.0.0.0", 8000, "default")
