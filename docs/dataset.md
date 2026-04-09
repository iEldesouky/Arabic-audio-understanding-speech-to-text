# Dataset Description

## Recommended primary dataset

- Mozilla Common Voice Arabic
  - https://commonvoice.mozilla.org/en
  - https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

## Additional datasets

- Arabic Speech Corpus: https://en.arabicspeechcorpus.com/
- MASC Arabic Speech Dataset: https://huggingface.co/datasets/hirundo-io/MASC
- Arabic Broadcast News Dataset: https://catalog.ldc.upenn.edu/LDC2006S46
- EJUST (private distribution)

## Local data layout

- data/raw: original audio and metadata
- data/processed: prepared manifests and cleaned metadata
- data/models: saved checkpoints
- data/index: retrieval artifacts

## Suggested split

- Train: 80%
- Validation: 10%
- Test: 10%

Use speaker-independent splits where possible.
