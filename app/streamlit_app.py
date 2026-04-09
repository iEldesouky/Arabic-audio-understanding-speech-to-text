from __future__ import annotations

from pathlib import Path
import tempfile
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
st.title("Deep Learning Arabic Audio Understanding and Retrieval")
st.caption("Speech -> Text -> Summary -> Search -> Speaker -> Emotion -> Keywords")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DATA_DIR = PROJECT_ROOT / "data"


def list_data_audio_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files)

with st.sidebar:
    st.header("Settings")
    asr_backend = st.selectbox("ASR backend", ["whisper", "wav2vec2"])
    asr_model = st.text_input(
        "ASR model",
        value="openai/whisper-small"
        if asr_backend == "whisper"
        else "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    )
    summary_model = st.text_input("Summary model", value="csebuetnlp/mT5_multilingual_XLSum")
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

input_mode = st.radio("Audio source", ["Upload file", "Use file from data folder"], horizontal=True)
uploaded = None
selected_data_file: Path | None = None
reference_text = st.text_area("Optional reference transcript (for WER)")
keywords_text = st.text_input("Keywords (comma-separated)", value="emergency, deadline, exam, طوارئ")

if input_mode == "Upload file":
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])
else:
    available_files = list_data_audio_files()
    if available_files:
        selected_display = st.selectbox(
            "Select an audio file from data/",
            options=available_files,
            format_func=lambda p: str(p.relative_to(PROJECT_ROOT)),
        )
        selected_data_file = selected_display
    else:
        st.warning("No audio files found under data/. Add files like .wav or .mp3 and refresh.")

if uploaded is not None:
    st.audio(uploaded)
elif selected_data_file is not None:
    st.audio(str(selected_data_file))

if st.button("Run analysis", type="primary"):
    if input_mode == "Upload file" and uploaded is None:
        st.warning("Please upload an audio file first.")
        st.stop()
    if input_mode == "Use file from data folder" and selected_data_file is None:
        st.warning("Please select a file from data/ first.")
        st.stop()

    custom_keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
    audio_path: str

    if input_mode == "Upload file":
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name
    else:
        audio_path = str(selected_data_file)

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config.models.asr_backend = asr_backend
    if asr_backend == "whisper":
        config.models.whisper_model = asr_model
    else:
        config.models.wav2vec_model = asr_model
    config.models.summary_model = summary_model
    config.models.emotion_model = emotion_model
    config.pipeline.speaker_count = int(n_speakers)
    config.pipeline.chunk_seconds = float(chunk_seconds)

    with st.spinner("Running full audio understanding pipeline..."):
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
            results = pipeline.search_engine.search(query, top_k=5) if pipeline.search_engine else []
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
