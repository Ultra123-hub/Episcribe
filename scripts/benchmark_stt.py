"""
STT benchmark: Sahara 2.5 vs faster-whisper vs MMS (facebook/mms-1b-all),
on real code-switched clinical audio from intronhealth/AfriSwitchCare.

WER/CER methodology mirrors the hackathon's own benchmarking reference
(github.com/intron-innovation/Intron-Multimodal-Benchmarking,
scripts/evaluations.py): jiwer for edit-distance metrics, plus a
"normalized" pass through whisper_normalizer's BasicTextNormalizer
(lowercase, strip punctuation/diacritics, collapse whitespace) since our
languages are non-English/code-switched, not the EnglishTextNormalizer
branch that script uses for pure English.

Usage:
    python scripts/benchmark_stt.py results.json

results.json shape: {
  "<language>": {
    "reference": "<ground truth transcript>",
    "duration_s": <float>,
    "models": {
      "sahara": {"hypothesis": "...", "elapsed_s": <float>},
      "whisper": {"hypothesis": "...", "elapsed_s": <float>},
      "mms": {"hypothesis": "...", "elapsed_s": <float>}
    }
  }, ...
}

Prints normalized + unnormalized WER/CER and latency tables (markdown).
"""
import json
import sys

import jiwer
from whisper_normalizer.basic import BasicTextNormalizer

_normalizer = BasicTextNormalizer(remove_diacritics=True)


def _clean(text: str) -> str:
    """Same shape as the Intron benchmark's clean_multilingual_text for
    non-English text: strip sentence punctuation Intron folds into
    silence markers, then run the Whisper-style basic normalizer
    (lowercase, strip remaining punctuation/diacritics, collapse
    whitespace). Empty results are guarded against (jiwer errors on an
    all-empty reference/hypothesis pair) the same way Intron's script
    guards it -- substitute a placeholder token rather than crash."""
    text = (text or "")
    for token in (",", ".", ":", ";", "?", "!", "(", ")", "[", "]"):
        text = text.replace(token, " ")
    text = " ".join(text.split())
    text = _normalizer(text)
    return text if text.strip() else "abcxyz"


def score(reference: str, hypothesis: str) -> dict:
    ref_unnorm = reference.strip() or "abcxyz"
    hyp_unnorm = hypothesis.strip() or "abcxyz"
    ref_norm = _clean(reference)
    hyp_norm = _clean(hypothesis)
    return {
        "unnormalized_wer": jiwer.wer(ref_unnorm, hyp_unnorm),
        "unnormalized_cer": jiwer.cer(ref_unnorm, hyp_unnorm),
        "normalized_wer": jiwer.wer(ref_norm, hyp_norm),
        "normalized_cer": jiwer.cer(ref_norm, hyp_norm),
    }


def main(results_path: str):
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    languages = sorted(data.keys())
    all_models = sorted({m for lang in data.values() for m in lang["models"]})

    print("## Normalized WER (lower is better)\n")
    print("| Language | " + " | ".join(all_models) + " |")
    print("|---|" + "---|" * len(all_models))
    wer_rows = {}
    for lang in languages:
        row = []
        for model in all_models:
            m = data[lang]["models"].get(model)
            if not m:
                row.append("—")
                continue
            s = score(data[lang]["reference"], m["hypothesis"])
            wer_rows.setdefault(model, {})[lang] = s
            row.append(f"{s['normalized_wer']*100:.1f}%")
        print(f"| {lang.title()} | " + " | ".join(row) + " |")

    print("\n## Normalized CER (lower is better)\n")
    print("| Language | " + " | ".join(all_models) + " |")
    print("|---|" + "---|" * len(all_models))
    for lang in languages:
        row = []
        for model in all_models:
            s = wer_rows.get(model, {}).get(lang)
            row.append(f"{s['normalized_cer']*100:.1f}%" if s else "—")
        print(f"| {lang.title()} | " + " | ".join(row) + " |")

    print("\n## Unnormalized WER (raw, no text normalization)\n")
    print("| Language | " + " | ".join(all_models) + " |")
    print("|---|" + "---|" * len(all_models))
    for lang in languages:
        row = []
        for model in all_models:
            s = wer_rows.get(model, {}).get(lang)
            row.append(f"{s['unnormalized_wer']*100:.1f}%" if s else "—")
        print(f"| {lang.title()} | " + " | ".join(row) + " |")

    print("\n## Latency (seconds, wall-clock per full clip)\n")
    print("| Language | Duration (s) | " + " | ".join(all_models) + " |")
    print("|---|---|" + "---|" * len(all_models))
    for lang in languages:
        row = []
        for model in all_models:
            m = data[lang]["models"].get(model)
            row.append(f"{m['elapsed_s']:.1f}" if m else "—")
        dur = data[lang].get("duration_s", "—")
        print(f"| {lang.title()} | {dur} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json")
