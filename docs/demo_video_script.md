# Demo video script (task 8)

Target: 2-3 minutes, 2-3 language pairs (per project scope — don't try to
cover everything). Public or unlisted YouTube upload.

**Before recording:** confirm `.env` has `EPISCRIBE_STT_BACKEND=sahara`
and a valid `SAHARA_API_KEY` set, so the demo actually uses Sahara 2.5
(not the offline Whisper fallback). Run `python app.py` and open the
local URL. Have this script open on a second screen/monitor.

## Beat 1 — Cold open (10-15s)

State the problem in one breath: *"Frontline health workers in West
Africa often speak in code-switched languages — mixing English with
Hausa, Yoruba, Pidgin. EpiScribe turns that speech directly into a
structured WHO IDSR case record, using Sahara 2.5."*

## Beat 2 — Language pair 1: Nigerian Pidgin-English (45-60s)

On the **New Consultation** tab:
1. Set language hint to "Nigerian Pidgin" (or leave Auto-detect).
2. Click record, speak a short code-switched narrative, e.g.:
   > *"Good morning doctor, my pikin get fever since two days, e dey
   > hot well well, e no wan chop, na so so vomiting since yesterday."*
3. Click **Transcribe audio into narrative** — show the transcript
   appearing (call out that this round-trips through Sahara 2.5).
4. Click **Generate clinical documentation** — show the resulting IDSR
   record: syndrome category, severity, ICD-10 code, reportable flag.

## Beat 3 — Language pair 2: Hausa-English (45-60s)

Same flow, different language, e.g.:
> *"Sannu likita, yarona yana da zazzabi tun kwana biyu, kuma yana da
> tari sosai, he's been coughing a lot and refusing food."*

Point out the SOAP note view alongside the IDSR fields — same encounter,
two clinical-note formats from one extraction call.

## Beat 4 — Spoken assistant reply (20-30s)

Switch to the **Assistant** tab, ask a short IDSR question, e.g.
*"What distinguishes acute watery diarrhea from acute bloody diarrhea
under IDSR?"* — let the spoken reply play (Sahara 2.5 streaming TTS)
alongside the text answer.

## Beat 5 — Close (15-20s)

One sentence each:
- *"Speech-to-text defaults to fully offline, and Sahara 2.5 is opt-in —
  see our privacy note for exactly what leaves the device."*
- *"Full benchmark comparing Sahara 2.5 against faster-whisper and MMS
  is in the repo, along with our STT and extraction-pipeline
  verification logs."*
- End on the repo URL / GitHub link.

## Notes

- If live code-switched speech feels unnatural to perform on camera,
  it's fine to read one of the real AfriSwitchCare example lines
  instead (see `benchmark/results.json` for real reference transcripts
  per language) — just say so in the video rather than presenting it as
  spontaneous.
- Keep each generation step visible on screen for a few seconds so
  viewers can actually read the output — don't cut away immediately.
- If extraction takes a while on your hardware (CPU-only inference can
  take a few minutes — see docs/extraction_verification.md), either
  record it and speed up that segment in editing, or narrate over it
  rather than sitting in silence.
