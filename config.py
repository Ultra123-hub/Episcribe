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
# Server binding
# ---------------------------------------------------------------------------
# Defaults to localhost-only. Set EPISCRIBE_SERVER_NAME=0.0.0.0 to accept
# connections from other devices on the same network (e.g. to open the app
# from a phone browser) — note this app has no authentication, so anyone
# on the same network/WiFi can reach it while it's bound this way.
#
# Hugging Face Spaces sets SPACE_ID automatically inside its containers.
# .env doesn't exist there (it's gitignored, never deployed), so without
# this the app would fall back to 127.0.0.1 — unreachable from outside the
# container, i.e. the Space would look broken to every visitor. Detect
# Spaces and default to 0.0.0.0 there instead; EPISCRIBE_SERVER_NAME still
# overrides if explicitly set. Also used elsewhere (app.py) to hide the
# Records tab on a public Space — visitors there would otherwise share one
# database and be able to read/export each other's submitted narratives.
ON_HF_SPACES = bool(os.getenv("SPACE_ID"))
SERVER_NAME = os.getenv("EPISCRIBE_SERVER_NAME", "0.0.0.0" if ON_HF_SPACES else "127.0.0.1")
SERVER_PORT = int(os.getenv("EPISCRIBE_SERVER_PORT", "7860"))

# Basic login prompt for demo.launch(auth=...). Only enabled when both are
# set — leaving either unset keeps the app open with no login, same as
# before. Matters most when SERVER_NAME is opened up beyond localhost.
AUTH_USERNAME = os.getenv("EPISCRIBE_AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("EPISCRIBE_AUTH_PASSWORD", "")

# Self-signed HTTPS. Browsers only allow microphone access (getUserMedia)
# on a secure context — HTTPS, or "localhost" specifically — so a phone
# opening this over plain http://<LAN-IP> can't use the mic at all. Only
# enabled when both cert files are set; the browser will show a
# self-signed-certificate warning to click through on first visit.
SSL_CERTFILE = os.getenv("EPISCRIBE_SSL_CERTFILE", "")
SSL_KEYFILE = os.getenv("EPISCRIBE_SSL_KEYFILE", "")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "episcribe.db"

# ---------------------------------------------------------------------------
# Extraction reliability
# ---------------------------------------------------------------------------
MAX_EXTRACTION_RETRIES = 3
LOW_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# STT backend selection + Sahara (Intron Voice API)
# ---------------------------------------------------------------------------
STT_BACKEND = os.getenv("EPISCRIBE_STT_BACKEND", "whisper").strip().lower()
SAHARA_API_KEY = os.getenv("SAHARA_API_KEY", "")
MAX_TOKENS_EXTRACTION = 1536
