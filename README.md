# MLOps Assignment 2 — Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment

**Course:** MLOps | PGD AI Program | IIT Jodhpur  
**Author:** Atreyee Halder (G25AIT2023)

---

## Project Description

This project demonstrates a complete MLOps pipeline for fine-tuning a **DistilBERT** model on the [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html#datasets) dataset to classify book reviews into **8 genres**. The workflow covers experiment tracking with Weights & Biases, model hosting on Hugging Face Hub, and reproducible training on Kaggle with GPU acceleration.

### Genre Labels

| # | Genre |
|---|-------|
| 1 | poetry |
| 2 | children |
| 3 | comics_graphic |
| 4 | fantasy_paranormal |
| 5 | history_biography |
| 6 | mystery_thriller_crime |
| 7 | romance |
| 8 | young_adult |

---

## Pipeline Overview

```
Data Ingestion → Baseline Model → Tokenization → Fine-Tuning → Evaluation → Deployment
     |                |                |              |              |            |
  UCSD Goodreads   TF-IDF +       DistilBERT     HuggingFace    W&B Logging   HF Hub +
  (8 genres)       LogReg          Tokenizer      Trainer        & Artifacts   GitHub
```

### Step-by-Step Workflow

1. **Data Collection** - Stream and sample 2,000 reviews per genre (16,000 total) from UCSD Goodreads hosted datasets.
2. **Train/Test Split** - 800 train + 200 test per genre → 6,400 training / 1,600 test samples.
3. **Baseline Model** - TF-IDF + Logistic Regression to establish a baseline performance benchmark.
4. **Tokenization** - Encode texts using `DistilBertTokenizerFast` with truncation and padding (max 512 tokens).
5. **Fine-Tuning** - Fine-tune `distilbert-base-cased` using HuggingFace `Trainer` with W&B integration.
6. **Evaluation** - Generate classification report, log metrics (accuracy, F1, loss) to W&B.
7. **Artifact Logging** - Save `eval_report.json` as a versioned W&B Artifact.
8. **Model Deployment** - Push fine-tuned model and tokenizer to Hugging Face Hub.
9. **Model Card** - Auto-update HF model card with evaluation results and hyperparameters.
10. **Version Control** - Push notebook, eval report, and README to GitHub.

----

## Model & Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-cased` |
| Max Token Length | 512 |
| Epochs | 3 |
| Train Batch Size | 16 |
| Eval Batch Size | 32 |
| Learning Rate | 3e-5 |
| Warmup Steps | 100 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Logging Steps | 50 |
| Eval Strategy | Per Epoch |
| Platform | Kaggle (GPU T4 x2) |

---

## Results 

| Metric | Score |
|--------|-------|
| Accuracy | 0.5919 |
| Weighted F1 | 0.5935 |
| Eval Loss | 2.2633 |

### Per-Genre Classification Report

| Genre | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| children | 0.65 | 0.65 | 0.65 | 200 |
| comics_graphic | 0.85 | 0.80 | 0.82 | 200 |
| fantasy_paranormal | 0.38 | 0.44 | 0.41 | 200 |
| history_biography | 0.54 | 0.55 | 0.54 | 200 |
| mystery_thriller_crime | 0.52 | 0.55 | 0.53 | 200 |
| poetry | 0.79 | 0.79 | 0.79 | 200 |
| romance | 0.68 | 0.66 | 0.67 | 200 |
| young_adult | 0.35 | 0.32 | 0.33 | 200 |

### Key Observations

- **Best performing genres:** `comics_graphic` and `poetry` - these have distinctive review language.
- **Most confused genres:** `young_adult` and `fantasy_paranormal` - frequently misclassified as `mystery_thriller_crime` due to overlapping vocabulary.
- **Overall:** 59.2% accuracy across 8 classes is well above the ~12.5% random baseline.

---

## MLOps Tools & Integration

| Tool | Purpose |
|------|---------|
| **Weights & Biases** | Experiment tracking, metric logging, artifact versioning |
| **Hugging Face Hub** | Model hosting, model card, inference API |
| **Kaggle** | Training platform with GPU T4 x2 acceleration |
| **GitHub** | Version control for notebook, eval report, and README |

### W&B Tracking Details

- **Project:** `mlops-assignment2`
- **Run Name:** `distilbert-run-1`
- **Logged Metrics:** `eval/loss`, `eval/accuracy`, `eval/f1`, `final/loss`, `final/accuracy`, `final/f1`, `train/loss`
- **Artifacts:** `eval-report` (classification report JSON)
- **System Metrics:** GPU utilization, memory, CPU, temperature

---

## Repository Structure

```
MLOPS-ASSIGNMENT2_Kaggle/
├── README.md                         # Project documentation (auto-generated + static)
├── mlops-assignment-kaggle.ipynb     # Full training notebook (Kaggle)
├── eval_report.json                  # Per-genre evaluation metrics (W&B artifact)
└── requirements.txt                  # Python dependencies
```

---

## Setup Instructions

1. Import the notebook into **Kaggle**.
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Enable Internet: **Settings → Internet → ON**
4. Add Kaggle Secrets:
   - `WANDB_API_KEY` - from [wandb.ai/authorize](https://wandb.ai/authorize)
   - `HF_TOKEN` - from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - `Kaggle-Push` - GitHub personal access token for repo push
5. Click **Run All**.

---

## Inference Example

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Atreyee-Halder/distilbert-goodreads-genres",
    top_k=None
)

result = classifier("A thrilling mystery set in Victorian London with unexpected twists.")
print(result)
```

---

## Links

- **GitHub:** [halderatreyee-hash/MLOPS-ASSIGNMENT2_Kaggle](https://github.com/halderatreyee-hash/MLOPS-ASSIGNMENT2_Kaggle)
- **Kaggle:** [mlops-assignment-kaggle](https://www.kaggle.com/code/ahalderg25ait2023/mlops-assignment-kaggle)
- **Hugging Face:** [Atreyee-Halder/distilbert-goodreads-genres](https://huggingface.co/Atreyee-Halder/distilbert-goodreads-genres)
- **W&B Dashboard:** [mlops-assignment2](https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment2)
