# Arabic Audio Understanding (Simplified)

This project is now intentionally minimal.

Flow in the interface:
1. Upload an audio file or choose one from `data/`
2. Play audio in Streamlit
3. Generate transcript with Whisper
4. Run simple text analysis (word stats + keyword hits)

## Project Structure
```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── index/
├── src/
│   └── audio_understanding/
│       ├── asr/
│       │   ├── __init__.py
│       │   └── whisper_asr.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── text.py
│       ├── __init__.py
│       └── simple_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Run
```bash
cd "/home/ibrahimeldesouky/Programming/projects/Arabic audio understanding-speech to text"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
streamlit run app/streamlit_app.py
```
