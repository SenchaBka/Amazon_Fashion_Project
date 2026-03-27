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

## Run Phase 2 Step 3 and Step 4

The project now includes separate runners for:

- Step 3: preprocessing + TF-IDF text representation
- Step 4: Logistic Regression training/tuning with a stratified 70/30 split

### Prerequisite

Step 3 expects the Step 1 subset CSV to exist at:

```text
outputs/phase 2/step 1/step1_subset_labeled.csv
```

If you have not created Step 1 artifacts yet, run:

```bash
python -m src.phase2.run_phase2_steps12 --data_path data/AMAZON_FASHION.json --subset_size 2000 --seed 42
```

### Step 3: TF-IDF representation

Default run:

```bash
python -m src.phase2.run_phase2_step3
```

Custom example:

```bash
python -m src.phase2.run_phase2_step3 \
  --subset_csv "outputs/phase 2/step 1/step1_subset_labeled.csv" \
  --out_dir "outputs/phase 2/step 3" \
  --max_features 20000 \
  --min_df 2 \
  --max_df 0.95 \
  --ngram_max 2
```

Step 3 outputs (default `outputs/phase 2/step 3/`):

- `step3_tfidf_matrix.npz`
- `step3_labels.csv`
- `step3_tfidf_vectorizer.joblib`
- `step3_vocabulary.csv`
- `step3_summary.json`

### Step 4: Logistic Regression training/tuning

Default run:

```bash
python -m src.phase2.run_phase2_step4
```

Custom example:

```bash
python -m src.phase2.run_phase2_step4 \
  --step3_dir "outputs/phase 2/step 3" \
  --out_dir "outputs/phase 2/step 4" \
  --seed 42 \
  --cv_folds 5
```

Step 4 outputs (default `outputs/phase 2/step 4/`):

- `step4_logreg_model.joblib`
- `step4_best_params.json`
- `step4_metrics.json`
- `step4_confusion_matrix.csv`
- `step4_test_predictions.csv`
- `step4_cv_results_top10.csv`
- `step4_summary.json`

### Recommended order

```bash
python -m src.phase2.run_phase2_step3
python -m src.phase2.run_phase2_step4
```

![bean](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMG43cDJ4eDV6NWc3MTBtM2M4MXpxZXl1YnludTh0MmU2aHNmc2pnaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RLcQGYmQU36d3FceiP/giphy.gif)
