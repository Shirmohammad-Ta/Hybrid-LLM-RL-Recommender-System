#  LLM-RL Recommender System

This project implements a hybrid recommender system that combines **Large Language Models (LLMs)** and **Reinforcement Learning (RL)** with a **self-regulation mechanism** to improve recommendation quality under data sparsity.

---

##  Key Features

-  LLM-based synthetic data generation (e.g., GPT-2)
-  Reinforcement Learning for adaptive recommendation (PPO)
-  Self-regulation to filter and prioritize high-quality synthetic data
-  Works in sparse data or cold-start scenarios
-  Evaluated on MovieLens and Amazon Books datasets

---


 

```bash
pip install transformers torch pandas scikit-learn stable-baselines3

## Run Preprocessing: python data/preprocess.py
## Train the Model: python src/train.py

## Datasets
MovieLens 1M: https://grouplens.org/datasets/movielens/
Amazon Reviews (Books): https://nijianmo.github.io/amazon/index.html

## Reproducibility:
Random seed fixed at 42
Scripts run end-to-end with one command
Training reproducible using same hyperparameters

