"""
EpiScribe configuration.

All paths are relative to the project root so the app runs the same way
whether launched from VS Code, a terminal, or a Hugging Face Space.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM (EpiCast fine-tuned MedGemma 4B, GGUF, Q4_K_M — Apache-2.0)
# ---------------------------------------------------------------------------
HF_MODEL_REPO = os.getenv("EPISCRIBE_MODEL_REPO", "Janeodum/epicast-medgemma-4b-gguf")
HF_MODEL_FILENAME = os.getenv("EPISCRIBE_MODEL_FILE", "medgemma-4b-epicast-Q4_K_M.gguf")
LOCAL_MODEL_PATH = MODELS_DIR / HF_MODEL_FILENAME

# Allow a fully custom local model path to override the Hub download
# (set EPISCRIBE_LOCAL_MODEL_PATH to an absolute .gguf path to use it instead)
CUSTOM_MODEL_PATH = os.getenv("EPISCRIBE_LOCAL_MODEL_PATH", "")

N_CTX = int(os.getenv("EPISCRIBE_N_CTX", "4096"))
N_THREADS = int(os.getenv("EPISCRIBE_N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.getenv("EPISCRIBE_N_GPU_LAYERS", "0"))  # 0 = CPU-only by default
MAX_TOKENS_EXTRACTION = 512
MAX_TOKENS_CHAT = 400
TEMPERATURE_EXTRACTION = 0.1   # low temp: we want consistent structured output
TEMPERATURE_CHAT = 0.4

# ---------------------------------------------------------------------------
# Offline speech-to-text (faster-whisper)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("EPISCRIBE_WHISPER_SIZE", "small")  # tiny/base/small/medium
WHISPER_DEVICE = os.getenv("EPISCRIBE_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("EPISCRIBE_WHISPER_COMPUTE", "int8")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "episcribe.db"

# ---------------------------------------------------------------------------
# Extraction reliability
# ---------------------------------------------------------------------------
MAX_EXTRACTION_RETRIES = 3
LOW_CONFIDENCE_THRESHOLD = 0.5
