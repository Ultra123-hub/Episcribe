"""
Loads the EpiCast fine-tuned MedGemma 4B GGUF model via llama-cpp-python.

Auto-downloads the weights from the Hugging Face Hub on first run
(Janeodum/epicast-medgemma-4b-gguf, Apache-2.0, ~2.49GB) and caches them
under ./models/. Set EPISCRIBE_LOCAL_MODEL_PATH to point at a different
.gguf file instead (see config.py).

This module is a lazy singleton: the model loads once, the first time
either the extraction agent or the chat agent needs it, and is reused
for both — this is the "one model, two jobs" decision from the concept
note that keeps memory/hosting footprint low.
"""
import sys
import threading

import config

_model = None
_lock = threading.Lock()


def _download_if_needed() -> str:
    if config.CUSTOM_MODEL_PATH:
        print(f"[model_manager] Using custom local model: {config.CUSTOM_MODEL_PATH}")
        return config.CUSTOM_MODEL_PATH

    if config.LOCAL_MODEL_PATH.exists():
        print(f"[model_manager] Found cached model at {config.LOCAL_MODEL_PATH}")
        return str(config.LOCAL_MODEL_PATH)

    print(
        f"[model_manager] Downloading {config.HF_MODEL_FILENAME} from "
        f"{config.HF_MODEL_REPO} (≈2.49GB, one-time download)..."
    )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.HF_MODEL_FILENAME,
        local_dir=str(config.MODELS_DIR),
    )
    print(f"[model_manager] Download complete: {path}")
    return path


def get_model():
    """Return the shared Llama instance, loading it on first call."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:  # re-check inside the lock
            return _model

        try:
            from llama_cpp import Llama
        except ImportError:
            print(
                "\n[model_manager] ERROR: llama-cpp-python is not installed.\n"
                "Install it with:  pip install llama-cpp-python\n"
                "On Windows, if that fails to build, try:\n"
                "  pip install llama-cpp-python --prefer-binary "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n",
                file=sys.stderr,
            )
            raise

        model_path = _download_if_needed()
        print("[model_manager] Loading model into memory (CPU)... this can take ~10-30s.")
        _model = Llama(
            model_path=model_path,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=config.N_GPU_LAYERS,
            chat_format=None,  # use the chat template embedded in the GGUF (Gemma format)
            verbose=False,
        )
        print("[model_manager] Model ready.")
        return _model


def chat(messages, max_tokens: int, temperature: float) -> str:
    """Thin wrapper around create_chat_completion returning just the text."""
    llm = get_model()
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result["choices"][0]["message"]["content"].strip()
