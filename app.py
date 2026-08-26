"""
EpiScribe — offline-capable multilingual clinical documentation assistant
for WHO IDSR reporting, with an integrated guidance chatbot.

Run locally:
    python app.py

The first run downloads the EpiCast fine-tuned MedGemma 4B GGUF model
(~2.49GB) and a faster-whisper model; both are cached locally afterward
and no further internet access is required.
"""
import gradio as gr
import pandas as pd

import config
from src import chat_agent, extraction_agent, storage, transcribe
from src.idsr_reference import SUPPORTED_LANGUAGES
from src.schema import EncounterRecord

storage.init_db()


# ---------------------------------------------------------------------------
# Tab 1: New Consultation
# ---------------------------------------------------------------------------
def do_transcribe(audio_path, language_hint):
    if not audio_path:
        return gr.update(), "No audio recorded."
    try:
        text, detected = transcribe.transcribe_audio(audio_path, language_hint)
    except Exception as exc:
        return gr.update(), f"Transcription failed: {exc}"
    if not text:
        return gr.update(), "Could not transcribe any speech from the recording."
    return text, f"Transcribed (detected language: {detected}). Review and edit before submitting."


def run_extraction(narrative, language_hint):
    result, error = extraction_agent.extract(narrative, language_hint)
    if result is None:
        return "", "❌ " + error, None, gr.update(visible=False)

    record = EncounterRecord(**result.model_dump(), raw_narrative=narrative.strip())
    record_id = storage.save_encounter(record)

    warn = ""
    if result.confidence < config.LOW_CONFIDENCE_THRESHOLD:
        warn = "\n\n⚠️ **Low confidence — please review this record manually.**"

    summary_md = f"""### ✅ Saved as Encounter #{record_id}

**Syndrome category:** {result.syndrome_category}
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
def refresh_records(keyword):
    rows = storage.list_encounters(limit=500, keyword=keyword or None)
    if not rows:
        return pd.DataFrame(columns=[
            "id", "timestamp", "syndrome_category", "severity",
            "reportable", "confidence", "summary",
        ])
    df = pd.DataFrame(rows)
    return df[["id", "timestamp", "syndrome_category", "severity",
               "reportable", "confidence", "summary"]]


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

        transcribe_btn.click(
            do_transcribe, inputs=[audio_input, language_hint],
            outputs=[narrative_box, transcribe_status],
        )
        submit_btn.click(
            run_extraction, inputs=[narrative_box, language_hint],
            outputs=[result_md, error_md, result_json, result_json],
        )

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
    demo.launch()
