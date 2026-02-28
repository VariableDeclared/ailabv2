#!/usr/bin/env python3
"""
Example client for interacting with the vLLM server.
"""

import requests
import json
from typing import List


class VLLMClient:
    """Client for interacting with vLLM Model Server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def get_server_info(self) -> dict:
        """Get server information"""
        response = requests.get(f"{self.base_url}/")
        return response.json()

    def list_available_models(self) -> dict:
        """List all available models from Hugging Face Hub"""
        response = requests.get(f"{self.base_url}/models/available")
        return response.json()

    def list_loaded_models(self) -> dict:
        """List all currently loaded models"""
        response = requests.get(f"{self.base_url}/models")
        return response.json()

    def load_model(self, model_name: str) -> dict:
        """Load a model from Hugging Face Hub"""
        response = requests.get(f"{self.base_url}/models/{model_name}")
        return response.json()

    def configure_model(self, model_name: str, **kwargs) -> dict:
        """Configure model parameters"""
        payload = {"model_name": model_name}
        payload.update(kwargs)
        response = requests.post(f"{self.base_url}/configure", json=payload)
        return response.json()

    def unload_model(self, model_name: str) -> dict:
        """Unload a loaded model"""
        response = requests.post(f"{self.base_url}/models/{model_name}/unload")
        return response.json()

    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using a loaded model"""
        payload = {"prompt": prompt}
        payload.update(kwargs)
        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json()

    def batch_load_models(self, model_names: List[str]) -> dict:
        """Load multiple models in parallel"""
        response = requests.post(f"{self.base_url}/models/batch_load", json=model_names)
        return response.json()


def main():
    """Example usage of the VLLM client"""
    client = VLLMClient()

    # Print server info
    print("=== Server Information ===")
    print(json.dumps(client.get_server_info(), indent=2))

    # List available models (this might not work without HF credentials)
    try:
        print("\n=== Available Models ===")
        available = client.list_available_models()
        print(json.dumps(available, indent=2)[:500] + "...")
    except Exception as e:
        print(f"Could not list available models: {e}")

    # List loaded models (initially empty)
    print("\n=== Loaded Models ===")
    loaded = client.list_loaded_models()
    print(json.dumps(loaded, indent=2))

    # Example: Load and configure a model
    model_name = "facebook/bart-large-cnn"
    print(f"\n=== Loading Model: {model_name} ===")
    load_result = client.load_model(model_name)
    print(json.dumps(load_result, indent=2))

    # Generate with a model
    print("\n=== Generating Text ===")
    prompt = "Write a short story about a robot learning to paint."
    generate_result = client.generate(prompt=prompt, temperature=0.8, max_tokens=100)
    print(json.dumps(generate_result, indent=2))

    # Print generated text
    if "generated_text" in generate_result:
        print(f"\nGenerated: {generate_result['generated_text']}")


if __name__ == "__main__":
    main()
