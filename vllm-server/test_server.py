#!/usr/bin/env python3
"""
Simple tests for the vLLM server.
Run these tests after starting the server.
"""

import requests
import json
import time


def test_server_info():
    """Test if server is running"""
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    print("✓ Server info endpoint works")
    return response.json()


def test_list_available_models():
    """Test listing available models"""
    response = requests.get("http://localhost:8000/models/available")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    print(f"✓ Found {len(data['available_models'])} available models")
    return data


def test_list_loaded_models():
    """Test listing loaded models"""
    response = requests.get("http://localhost:8000/models")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Currently loaded {data['count']} model(s)")
    return data


def test_configure_model():
    """Test configuring a model"""
    response = requests.post(
        "http://localhost:8000/configure",
        json={"model_name": "test-configured-model", "gpu_memory_utilization": 0.8},
    )
    assert response.status_code == 200
    print("✓ Model configuration works")
    return response.json()


def test_load_model(model_name: str):
    """Test loading a model"""
    print(f"  Attempting to load {model_name}...")
    response = requests.get(f"http://localhost:8000/models/{model_name}")

    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            print(f"  ✗ Loading failed: {data['error']}")
            return False
        print(f"  ✓ Model loading initiated")
        return True
    else:
        print(f"  ✗ Failed with status {response.status_code}")
        return False


def test_generate(prompt: str = "Hello, world!"):
    """Test text generation"""
    response = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt, "temperature": 0.7, "max_tokens": 50},
    )

    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Generated: {data.get('generated_text', 'N/A')[:50]}...")
        return True
    else:
        print(f"  ✗ Generation failed with status {response.status_code}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("vLLM Server Tests")
    print("=" * 50)

    # Test server info
    print("\n1. Testing server info...")
    test_server_info()

    # Test available models
    print("\n2. Testing available models list...")
    test_list_available_models()

    # Test loaded models
    print("\n3. Testing loaded models...")
    test_list_loaded_models()

    # Test model configuration
    print("\n4. Testing model configuration...")
    test_configure_model()

    # Test generation (requires loaded model)
    print("\n5. Testing text generation...")
    print("   Note: This may fail if no model is loaded yet.")
    test_generate()

    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
