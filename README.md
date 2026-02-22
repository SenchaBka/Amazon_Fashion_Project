# Amazon Fashion – COMP 262 Phase 1

Lexicon-based sentiment analysis (VADER + TextBlob) on the Amazon Fashion review dataset.

## Dataset

Place the raw JSONL file at:

```
data/AMAZON_FASHION.json
```

or `data/AMAZON_FASHION.jsonl` — the loader accepts both extensions.

Download: [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) → AMAZON_FASHION 5-core or full.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# Download TextBlob corpora (one-time)
python -m textblob.download_corpora
```

## Run Phase 1 Lexicon Pipeline

```bash
python -m src.run_phase1_lexicon --data_path data/AMAZON_FASHION.json --seed 42 --sample_size 1000
```

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data/AMAZON_FASHION.json` | Path to JSONL dataset |
| `--seed` | `42` | Random seed for sampling |
| `--sample_size` | `1000` | Number of reviews to sample |
| `--out_dir` | `outputs` | Directory for output files |

## Outputs

All files are written to `outputs/`:

| File | Description |
|---|---|
| `lexicon_choice.txt` | Justification for choosing VADER + TextBlob |
| `sample_1000.csv` | Sampled reviews with labels and predictions |
| `metrics.json` | Accuracy, precision, recall, F1 per model |
| `comparison.csv` | Side-by-side metrics table |
| `confusion_vader.csv` | 3×3 confusion matrix – VADER |
| `confusion_textblob.csv` | 3×3 confusion matrix – TextBlob |

## Source layout

```
src/
  io_utils.py            – JSONL loader
  preprocess_lexicon.py  – Labelling, cleaning, sampling
  lexicon_models.py      – VADER and TextBlob inference
  evaluate.py            – Metrics and output writers
  run_phase1_lexicon.py  – CLI entrypoint
```
