"""
EpiScribe — offline-capable multilingual clinical documentation assistant
for WHO IDSR reporting, with an integrated guidance chatbot.

Run locally:
    python app.py

The first run downloads the EpiCast fine-tuned MedGemma 4B GGUF model
(~2.49GB) and a faster-whisper model; both are cached locally afterward
and no further internet access is required (unless EPISCRIBE_STT_BACKEND
is set to "sahara", which requires connectivity — see src/transcribe.py).
"""
import sys

import gradio as gr
import pandas as pd

import config
from src import chat_agent, extraction_agent, storage, transcribe
from src.idsr_reference import SUPPORTED_LANGUAGES
from src.schema import EncounterRecord

# --- Satisfy HF Spaces' ZeroGPU startup check -------------------------------
# This Space's hardware is forced to ZeroGPU (CPU-basic downgrade requires a
# PRO subscription on this account, and CPU basic wasn't selectable when
# creating a fresh Space either). HF's ZeroGPU runtime refuses to start any
# app with no @spaces.GPU-decorated function at all ("No @spaces.GPU
# function detected during startup"). This app is CPU-only by design
# (llama-cpp-python, EPISCRIBE_N_GPU_LAYERS=0; faster-whisper on CPU) and
# never needs a GPU — this dummy function is never called, it exists purely
# to pass that check. Guarded so local runs (no `spaces` package installed,
# no GPU available anyway) are unaffected.
if config.ON_HF_SPACES:
    try:
        import spaces

        @spaces.GPU
        def _zerogpu_startup_check():  # pragma: no cover — never invoked
            pass
    except ImportError:
        pass
# -----------------------------------------------------------------------------

# --- Silence benign Windows asyncio connection-reset noise ------------------
# On Windows, asyncio's ProactorEventLoop logs an "Exception in callback"
# whenever a client (browser tab closing, phone locking/backgrounding,
# WiFi hiccup) forces a connection closed before the loop finishes its own
# cleanup — WinError 10054. Nothing is actually broken; no request fails,
# no data is lost. This is a well-known, longstanding Proactor quirk
# specific to Windows (doesn't happen on Linux/Mac), and gets noisier once
# less-reliable clients like mobile browsers are involved (MODE B). This
# patches just that one cleanup path to swallow ConnectionResetError
# instead of logging it — real errors elsewhere are untouched.
if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _quiet_call_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _quiet_call_connection_lost
# -----------------------------------------------------------------------------

# --- Workaround for a gradio_client 1.3.0 bug -------------------------------
# gradio_client.utils._json_schema_to_python_type assumes every JSON-schema
# node is a dict, but a schema can legitimately contain a bare boolean (e.g.
# {"additionalProperties": true}), which raises
# `TypeError: argument of type 'bool' is not iterable` and breaks every page
# load (the "/" route builds api_info on every request). Fixed upstream in
# later gradio_client releases, but 1.3.0 is what gradio==4.44.1 pins to, so
# we patch it at runtime instead of hand-editing site-packages (which a
# fresh `pip install` would wipe out). Safe to remove once gradio/gradio_client
# are upgraded past this bug.
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# -----------------------------------------------------------------------------

storage.init_db()

# Encounters logged with the same syndrome_category within this many days
# trigger the cluster banner in run_extraction below (see
# storage.recent_cluster_count for what this signal does and doesn't mean).
CLUSTER_WINDOW_DAYS = 7
CLUSTER_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Tab 1: New Consultation
# ---------------------------------------------------------------------------
def do_transcribe(audio_path, language_hint):
    if not audio_path:
        return gr.update(), "No audio recorded.", ""
    try:
        text, _backend_detected, audio_hash = transcribe.transcribe_audio(audio_path, language_hint)
    except Exception as exc:
        return gr.update(), f"Transcription failed: {exc}", ""
    if not text:
        return gr.update(), "Could not transcribe any speech from the recording.", ""
    # Language is resolved during extraction (one LLM call, see
    # extraction_agent.py) rather than here — an earlier version ran a
    # separate detect_language() pass immediately after transcription for
    # a more responsive display, but that cost a full extra LLM round-trip
    # per submission and measurably hurt CPU inference latency. Speed won.
    status = "Transcribed. Review and edit before submitting."
    return text, status, audio_hash


def run_extraction(narrative, language_hint, audio_hash):
    result, error = extraction_agent.extract(narrative, language_hint)
    if result is None:
        return "", "", "❌ " + error, None, gr.update(visible=False)

    record = EncounterRecord(
        **result.model_dump(), raw_narrative=narrative.strip(), audio_hash=audio_hash or ""
    )
    record_id = storage.save_encounter(record)
    cluster_count = storage.recent_cluster_count(result.syndrome_category, days=CLUSTER_WINDOW_DAYS)

    cluster_banner = ""
    if cluster_count >= CLUSTER_THRESHOLD:
        cluster_banner = (
            f"\n\n🚨 **{cluster_count} cases of {result.syndrome_category} logged in the "
            f"last {CLUSTER_WINDOW_DAYS} days** — possible cluster, consider investigating.\n"
        )

    low_confidence_warn = ""
    if result.confidence < config.LOW_CONFIDENCE_THRESHOLD:
        low_confidence_warn = "\n\n⚠️ **Low confidence — please review this record manually.**"

    idsr_md = f"""### ✅ Saved as Encounter #{record_id}
{cluster_banner}
**Syndrome category:** {result.syndrome_category}
**Public health category:** {", ".join(result.public_health_category) or "—"}
**Severity:** {result.severity}  |  **Onset:** {result.onset_days if result.onset_days is not None else "unknown"} day(s)
**Age group:** {result.age_group}  |  **Sex:** {result.sex}
**ICD-10:** {", ".join(result.icd10_codes) or "—"}
**Reportable (IDSR):** {"Yes" if result.reportable else "No"}
**Confidence:** {result.confidence:.2f}
**Language detected:** {result.language_detected}

**Summary:** {result.summary}
**Symptoms:** {", ".join(result.symptoms) or "—"}
{low_confidence_warn}
"""

    guard_banner = ""
    if result.hallucination_flags:
        flag_lines = "\n".join(f"- {f}" for f in result.hallucination_flags)
        guard_banner = (
            "\n> ⚠️ **Hallucination Guard flagged this note — verify before use:**\n"
            f"{flag_lines}\n"
        )

    soap_md = f"""### SOAP Note — Encounter #{record_id}
{guard_banner}
**Subjective**
{result.soap_subjective or "—"}

**Objective**
{result.soap_objective or "—"}

**Assessment**
{result.soap_assessment or "—"}

**Plan**
{result.soap_plan or "—"}
"""

    return idsr_md, soap_md, "", result.model_dump(), gr.update(visible=True)


# ---------------------------------------------------------------------------
# Tab 2: Records
# ---------------------------------------------------------------------------
_RECORD_COLUMNS = [
    "flag", "id", "timestamp", "syndrome_category", "public_health_category",
    "symptoms", "severity", "reportable", "confidence", "summary",
]


def refresh_records(keyword):
    rows = storage.list_encounters(limit=500, keyword=keyword or None)
    if not rows:
        return pd.DataFrame(columns=_RECORD_COLUMNS)
    df = pd.DataFrame(rows)
    df["symptoms"] = df["symptoms"].apply(lambda v: ", ".join(v))
    df["public_health_category"] = df["public_health_category"].apply(lambda v: ", ".join(v))
    # Visual flag for rows a supervisor should double-check: either a low
    # confidence score, or the Hallucination Guard finding something in the
    # SOAP note ungrounded in the extracted symptoms/narrative — either
    # signal is enough to warrant a manual look, so either one lights this up.
    has_guard_flags = df["hallucination_flags"].apply(lambda v: bool(v))
    low_confidence = df["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    df["flag"] = (low_confidence | has_guard_flags).apply(lambda x: "⚠️" if x else "")
    return df[_RECORD_COLUMNS]


def do_export_csv():
    path = storage.export_csv()
    return path


# ---------------------------------------------------------------------------
# Tab 3: Assistant
# ---------------------------------------------------------------------------
def chat_respond(message, history):
    history = history or []
    reply_text = chat_agent.reply(message, history)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply_text},
    ]
    return "", history


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="EpiScribe") as demo:
    gr.Markdown(
        "# 🩺 EpiScribe\n"
        "Offline-capable multilingual clinical documentation for WHO IDSR "
        "reporting, with an integrated guidance assistant."
    )

    with gr.Tab("New Consultation"):
        with gr.Row():
            with gr.Column(scale=2):
                language_hint = gr.Dropdown(
                    SUPPORTED_LANGUAGES, value="Auto-detect", label="Language hint"
                )
                narrative_box = gr.Textbox(
                    label="Consultation narrative",
                    placeholder='e.g. "Di pikin body dey hot, e dey cough, '
                                'and e get rash for body since two days."',
                    lines=5,
                )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"], type="filepath",
                    label="Or record/upload audio",
                )
                transcribe_btn = gr.Button("🎙️ Transcribe audio into narrative")
                transcribe_status = gr.Markdown("")
                submit_btn = gr.Button("Generate clinical documentation", variant="primary")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("IDSR Record"):
                        idsr_md = gr.Markdown(label="IDSR structured record")
                    with gr.Tab("SOAP Note"):
                        soap_md = gr.Markdown(label="SOAP note")
                error_md = gr.Markdown("")
                result_json = gr.JSON(label="Raw structured output", visible=False)

        # Carries the source audio's hash through to the saved record —
        # provenance only, not shown in the UI. Lets a later "why did two
        # recordings produce the same transcript?" question be answered
        # definitively instead of guessed at.
        audio_hash_state = gr.State(value="")

        transcribe_btn.click(
            do_transcribe, inputs=[audio_input, language_hint],
            outputs=[narrative_box, transcribe_status, audio_hash_state],
        )
        submit_btn.click(
            run_extraction,
            inputs=[narrative_box, language_hint, audio_hash_state],
            outputs=[idsr_md, soap_md, error_md, result_json, result_json],
        )

    # Hidden on a public HF Space: every visitor there shares one database,
    # so listing/exporting records would let anyone read what anyone else
    # submitted. Fully present for local/self-hosted use — see config.py.
    if not config.ON_HF_SPACES:
        with gr.Tab("Records"):
            with gr.Row():
                keyword_box = gr.Textbox(label="Filter by keyword (syndrome, summary, narrative)", scale=3)
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
                export_btn = gr.Button("⬇️ Export CSV", scale=1)
            gr.Markdown("⚠️ in the **flag** column marks encounters below the confidence threshold, or where the Hallucination Guard found SOAP-note content ungrounded in the extracted symptoms/narrative. Open the SOAP Note tab for that encounter to see the specific reason.")
            records_table = gr.Dataframe(interactive=False, wrap=True)
            export_file = gr.File(label="Exported CSV", visible=True)

            refresh_btn.click(refresh_records, inputs=[keyword_box], outputs=[records_table])
            export_btn.click(do_export_csv, outputs=[export_file])
            demo.load(refresh_records, inputs=[keyword_box], outputs=[records_table])

    with gr.Tab("Assistant"):
        gr.Markdown(
            "Ask about IDSR case definitions, reportability, or say "
            "*\"show my recent records\"* to pull up your last saved encounters."
        )
        chatbot = gr.Chatbot(type="messages", height=420)
        chat_input = gr.Textbox(label="Message", placeholder="e.g. What distinguishes AWD from ABD under IDSR?")
        chat_send = gr.Button("Send", variant="primary")

        chat_send.click(chat_respond, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat_respond, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])

if __name__ == "__main__":
    auth = (
        (config.AUTH_USERNAME, config.AUTH_PASSWORD)
        if config.AUTH_USERNAME and config.AUTH_PASSWORD
        else None
    )
    ssl_enabled = bool(config.SSL_CERTFILE and config.SSL_KEYFILE)
    demo.launch(share=True,
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        auth=auth,
        ssl_certfile=config.SSL_CERTFILE or None,
        ssl_keyfile=config.SSL_KEYFILE or None,
        # Skip verifying our own self-signed chain, but only when SSL is
        # actually in use — leave the default (True) alone otherwise.
        ssl_verify=not ssl_enabled,
    )
