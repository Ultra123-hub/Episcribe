"""
EpiScribe — offline-capable multilingual clinical documentation assistant
for WHO IDSR reporting, with an integrated guidance chatbot.

Run locally:
    python app.py

The first run downloads the EpiCast fine-tuned MedGemma 4B GGUF model
(~2.49GB) and a faster-whisper model; both are cached locally afterward
and no further internet access is required.
"""
import sys

import gradio as gr
import pandas as pd

import config
from src import chat_agent, extraction_agent, storage, transcribe
from src.idsr_reference import SUPPORTED_LANGUAGES
from src.schema import EncounterRecord

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


# ---------------------------------------------------------------------------
# Tab 1: New Consultation
# ---------------------------------------------------------------------------
def do_transcribe(audio_path, language_hint):
    if not audio_path:
        return gr.update(), "No audio recorded.", ""
    try:
        text, detected, audio_hash = transcribe.transcribe_audio(audio_path, language_hint)
    except Exception as exc:
        return gr.update(), f"Transcription failed: {exc}", ""
    if not text:
        return gr.update(), "Could not transcribe any speech from the recording.", ""
    return text, f"Transcribed (detected language: {detected}). Review and edit before submitting.", audio_hash


def run_extraction(narrative, language_hint, audio_hash):
    result, error = extraction_agent.extract(narrative, language_hint)
    if result is None:
        return "", "❌ " + error, None, gr.update(visible=False)

    record = EncounterRecord(
        **result.model_dump(), raw_narrative=narrative.strip(), audio_hash=audio_hash or ""
    )
    record_id = storage.save_encounter(record)

    warn = ""
    if result.confidence < config.LOW_CONFIDENCE_THRESHOLD:
        warn = "\n\n⚠️ **Low confidence — please review this record manually.**"

    summary_md = f"""### ✅ Saved as Encounter #{record_id}

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
{warn}
"""
    return summary_md, "", result.model_dump(), gr.update(visible=True)


# ---------------------------------------------------------------------------
# Tab 2: Records
# ---------------------------------------------------------------------------
_RECORD_COLUMNS = [
    "id", "timestamp", "syndrome_category", "public_health_category",
    "symptoms", "severity", "reportable", "confidence", "summary",
]


def refresh_records(keyword):
    rows = storage.list_encounters(limit=500, keyword=keyword or None)
    if not rows:
        return pd.DataFrame(columns=_RECORD_COLUMNS)
    df = pd.DataFrame(rows)
    df["symptoms"] = df["symptoms"].apply(lambda v: ", ".join(v))
    df["public_health_category"] = df["public_health_category"].apply(lambda v: ", ".join(v))
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
                    label="Or record/upload audio (offline transcription)",
                )
                transcribe_btn = gr.Button("🎙️ Transcribe audio into narrative")
                transcribe_status = gr.Markdown("")
                submit_btn = gr.Button("Extract structured IDSR record", variant="primary")
            with gr.Column(scale=2):
                result_md = gr.Markdown(label="Structured record")
                error_md = gr.Markdown("")
                result_json = gr.JSON(label="Raw structured output", visible=False)

        # Carries the source audio's hash from transcription through to the
        # saved record — provenance only, not shown in the UI. Lets a later
        # "why did two recordings produce the same transcript?" question be
        # answered definitively instead of guessed at.
        audio_hash_state = gr.State(value="")

        transcribe_btn.click(
            do_transcribe, inputs=[audio_input, language_hint],
            outputs=[narrative_box, transcribe_status, audio_hash_state],
        )
        submit_btn.click(
            run_extraction, inputs=[narrative_box, language_hint, audio_hash_state],
            outputs=[result_md, error_md, result_json, result_json],
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
    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        auth=auth,
        ssl_certfile=config.SSL_CERTFILE or None,
        ssl_keyfile=config.SSL_KEYFILE or None,
        # Skip verifying our own self-signed chain, but only when SSL is
        # actually in use — leave the default (True) alone otherwise.
        ssl_verify=not ssl_enabled,
    )
