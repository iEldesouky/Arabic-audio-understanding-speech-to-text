from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.config import load_config
from audio_understanding.pipeline import AudioUnderstandingPipeline
from audio_understanding.utils.metrics import compute_wer


st.set_page_config(page_title="Arabic Audio Understanding", layout="wide")
st.title("Deep Learning Arabic Audio Understanding")
st.caption("Speech -> Text -> Summary -> Search -> Speaker -> Emotion -> Keywords")

with st.sidebar:
    st.header("Settings")
    asr_backend = st.selectbox("ASR backend", ["whisper", "wav2vec2"])
    asr_model = st.text_input(
        "ASR model",
        value="openai/whisper-small"
        if asr_backend == "whisper"
        else "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    )
    summary_model = st.text_input(
        "Summary model",
        value="csebuetnlp/mT5_multilingual_XLSum",
    )
    emotion_model = st.text_input(
        "Emotion model",
        value="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    )
    n_speakers = st.number_input("Expected speakers", min_value=1, max_value=10, value=2)
    chunk_seconds = st.slider("Speaker chunk seconds", min_value=1.0, max_value=8.0, value=2.0)

    enable_summary = st.checkbox("Enable summary", value=True)
    enable_search = st.checkbox("Enable semantic search", value=True)
    enable_speaker = st.checkbox("Enable speaker identification", value=True)
    enable_emotion = st.checkbox("Enable emotion detection", value=True)

uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "m4a"])
reference_text = st.text_area("Optional reference transcript (for WER)")
keywords_text = st.text_input("Keywords (comma-separated)", value="طوارئ, deadline, exam")

if uploaded is not None:
    st.audio(uploaded)

if st.button("Run analysis", type="primary"):
    if uploaded is None:
        st.warning("Please upload an audio file first.")
        st.stop()

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config.models.asr_backend = asr_backend
    config.models.asr_model = asr_model
    config.models.summary_model = summary_model
    config.models.emotion_model = emotion_model
    config.pipeline.speaker_count = int(n_speakers)
    config.pipeline.chunk_seconds = float(chunk_seconds)

    custom_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        audio_path = tmp.name

    with st.spinner("Running deep learning pipeline..."):
        pipeline = AudioUnderstandingPipeline(config)
        output = pipeline.run(
            audio_path=audio_path,
            enable_summary=enable_summary,
            enable_search=enable_search,
            enable_speaker_id=enable_speaker,
            enable_emotion=enable_emotion,
            custom_keywords=custom_keywords,
        )

    st.subheader("Transcript")
    st.write(output.transcript)

    if reference_text.strip():
        wer_value = compute_wer(reference_text, output.transcript)
        st.metric("WER", f"{wer_value:.3f}")

    if output.summary:
        st.subheader("Summary")
        st.write(output.summary)

    st.subheader("Keyword Spotting")
    if output.keyword_hits:
        st.table([{"keyword": hit.keyword, "count": hit.count} for hit in output.keyword_hits])
    else:
        st.info("No keywords detected.")

    if enable_search:
        st.subheader("Semantic Search")
        query = st.text_input("Search query", value="ملخص")
        if query:
            results = pipeline.search_engine.search(query, top_k=5)
            if results:
                for idx, item in enumerate(results, start=1):
                    st.write(f"{idx}. ({item.score:.3f}) {item.text}")
            else:
                st.info("No indexed transcript chunks found.")

    if enable_speaker:
        st.subheader("Speaker Segments")
        if output.speaker_segments:
            st.table(
                [
                    {
                        "speaker": seg.speaker_label,
                        "start_sec": round(seg.start_sec, 2),
                        "end_sec": round(seg.end_sec, 2),
                    }
                    for seg in output.speaker_segments
                ]
            )
        else:
            st.info("No speaker segments detected.")

    if enable_emotion:
        st.subheader("Emotion Detection")
        if output.emotions:
            st.table([{"label": e.label, "score": round(e.score, 4)} for e in output.emotions])
        else:
            st.info("No emotion predictions available.")
