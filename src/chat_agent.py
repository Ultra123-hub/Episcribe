"""
Guidance chatbot. Reuses the same loaded model as the extraction agent
(no second model to host) with a different system prompt, and can pull
the clinician's own saved records into context when asked about them.

This is intentionally simple heuristic routing rather than a full
tool-calling agent loop — enough to satisfy "ask about case definitions"
and "show my recent records" without the complexity a 7-day build can't
absorb.
"""
import re

import config
from src import model_manager, storage
from src.idsr_reference import reference_block_for_prompt

SYSTEM_PROMPT = f"""You are EpiScribe Assistant, a concise clinical-surveillance \
helper for community health workers and clinicians using WHO IDSR. You answer \
questions about IDSR case definitions, reportability, and syndrome distinctions, \
and you can summarize the clinician's own previously logged encounters when \
given them as context. Keep answers short, practical, and point-of-care \
appropriate. If asked something outside IDSR/clinical documentation scope, \
say so briefly and redirect to what you can help with.

Reference WHO IDSR syndrome categories:
{reference_block_for_prompt()}"""

_RECORD_TRIGGERS = re.compile(
    r"\b(my (records?|encounters?|cases?)|recent (records?|cases?|encounters?)|"
    r"show (me )?(my )?(records?|cases?)|last \d+ (records?|cases?))\b",
    re.IGNORECASE,
)


def _maybe_records_context(user_message: str) -> str:
    if not _RECORD_TRIGGERS.search(user_message):
        return ""
    rows = storage.list_encounters(limit=5)
    if not rows:
        return "\n\n[No saved encounters found yet.]"
    lines = ["\n\n[Clinician's 5 most recent saved encounters:]"]
    for r in rows:
        lines.append(
            f"- #{r['id']} ({r['timestamp']}): {r['syndrome_category']}, "
            f"severity={r['severity']}, reportable={r['reportable']}, "
            f"summary: {r['summary']}"
        )
    return "\n".join(lines)


def reply(user_message: str, history: list) -> str:
    """
    history: list of {"role": "user"/"assistant", "content": str} from prior turns
    (Gradio's gr.Chatbot "messages" format).
    """
    context = _maybe_records_context(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # keep the last few turns only, to stay within context window comfortably
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_message + context})

    try:
        return model_manager.chat(
            messages=messages,
            max_tokens=config.MAX_TOKENS_CHAT,
            temperature=config.TEMPERATURE_CHAT,
        )
    except Exception as exc:
        return f"Sorry, the assistant hit an error: {exc}"
