# Dataset Description

## Primary Dataset (Recommended)
- Mozilla Common Voice Arabic
- URL: https://commonvoice.mozilla.org/en
- HF loader used in this project: `mozilla-foundation/common_voice_17_0`, subset `ar`

## Additional Arabic Datasets
1. Arabic Speech Corpus: https://en.arabicspeechcorpus.com/
2. MASC Arabic Speech Dataset: https://huggingface.co/datasets/hirundo-io/MASC
3. Arabic Broadcast News Dataset: https://catalog.ldc.upenn.edu/LDC2006S46
4. EJUST dataset (private): https://drive.google.com/drive/folders/0B28Rhpdi0XLMfldKUUN6ZlhCTmRwLUhaUTZBN3ZBMUFDcjRVWVV3TTFtVTZobUVhMFBGSzg?resourcekey=0-x-kF6vPagWEyH9kjdVSl_g

## Why Common Voice for This Project
- Open and easy to load programmatically
- Contains Arabic utterances with aligned transcripts
- Good baseline dataset for reproducible student experiments

## Data Pipeline in This Repo
1. Download split and audio clips using `scripts/download_common_voice.py`
2. Build ASR manifest with `scripts/prepare_data.py`
3. Train/evaluate ASR and run end-to-end pipeline
