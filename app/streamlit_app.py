from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.simple_pipeline import SimpleAudioPipeline


st.set_page_config(page_title="Arabic Speech To Text", layout="wide")
st.title("Arabic Audio Understanding")
st.caption("Upload or pick an audio file, then transcribe and analyze it.")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DATA_DIR = PROJECT_ROOT / "data"


def list_data_audio_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files)

with st.sidebar:
    st.header("Options")
    model_name = st.text_input("Whisper model", value="openai/whisper-small")
    keywords_text = st.text_input("Keywords (comma-separated)", value="طوارئ, امتحان, موعد")

input_mode = st.radio("Audio source", ["Upload file", "Use file from data folder"], horizontal=True)
uploaded = None
selected_data_file: Path | None = None

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

    with st.spinner("Transcribing and analyzing..."):
        pipeline = SimpleAudioPipeline(model_name=model_name)
        output = pipeline.process(audio_path=audio_path, keywords=custom_keywords)

    st.subheader("Transcript")
    st.write(output.transcript)

    st.subheader("Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration (sec)", f"{output.duration_sec:.2f}")
    c2.metric("Words", str(output.word_count))
    c3.metric("Unique words", str(output.unique_word_count))

    st.write("Top words")
    if output.top_words:
        st.table([{"word": w, "count": c} for w, c in output.top_words])
    else:
        st.info("No words found in transcript.")

    st.write("Keyword hits")
    if output.keyword_hits:
        st.table([{"keyword": k, "count": c} for k, c in output.keyword_hits.items()])
    else:
        st.info("No keywords were detected.")
