# MLOps Assignment 2 — Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment

**Course:** MLOps | PGD AI Program | IIT Jodhpur  
**Author:** Atreyee Halder (G25AIT2023)

## Project Description

This project demonstrates a complete MLOps pipeline for fine-tuning a DistilBERT model on the UCSD Goodreads book review dataset to classify reviews into 8 genres. The workflow covers experiment tracking with Weights & Biases, model hosting on Hugging Face Hub, and reproducible training on Kaggle with GPU acceleration.

## Setup Instructions

1. Import the notebook into Kaggle
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Enable Internet: Settings → Internet → ON
4. Add secrets: WANDB_API_KEY, HF_TOKEN, GIT_WHOLE
5. Click Run All

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.5719 |
| F1 Score  | 0.5719 |
| Eval Loss | 2.3015 |

## Links

- **GitHub:** https://github.com/halderatreyee-hash/MLOPS-ASSIGNMENT2_Kaggle
- **Kaggle:** https://www.kaggle.com/code/ahalderg25ait2023/mlops-assignment-kaggle
- **Hugging Face:** https://huggingface.co/Atreyee-Halder/distilbert-goodreads-genres
- **W&B:** https://wandb.ai/g25ait2023-iit-jodhpur/mlops-assignment2
