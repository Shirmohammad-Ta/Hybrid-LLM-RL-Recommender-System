#  Hybrid LLM-RL Recommender System

This project implements a novel hybrid recommender system that combines **Large Language Models (LLMs)** with **Reinforcement Learning (RL)** and a **Self-Regulation Mechanism** to improve recommendation quality, especially in sparse-data environments.

---

##  Key Features

-  Synthetic data generation using GPT-based LLMs
-  Reinforcement Learning with PPO for feedback-driven optimization
-  Self-regulation module to select high-quality, diverse training samples
-  Evaluation on MovieLens and Amazon Books datasets
-  Designed for cold-start and low-data scenarios

---

##  Project Structure

```
Hybrid-LLM-RL-Recommender-System/
│
├── data/
│   ├── movielens_sample.csv
│   ├── amazon_books_sample.csv
│   ├── preprocess.py
│   └── processed/
│
├── src/
│   ├── llm_generator.py
│   ├── rl_agent.py
│   ├── self_regulation.py
│   ├── train.py
│   └── utils.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── plot_results.py
│   ├── RESULTS.md
│   └── README.md
│
├── config.yaml
└── README.md  ← (this file)
```


---

##  Installation

```bash
pip install torch transformers scikit-learn stable-baselines3 matplotlib pandas

---

##  How to Run

###  Preprocess the data

```bash
python data/preprocess.py


###  Train the system

```bash
python src/train.py


---

##  Evaluation

### Evaluation metrics include:

- Precision@10
- Recall@10
- NDCG@10

### Visualization tools available in:

- `evaluation1/plot_results.py`

### Summary results:

- `evaluation1/RESULTS.md`

---

##  Datasets

- [MovieLens 1M](https://grouplens.org/datasets/movielens/)
- [Amazon Reviews (Books)](https://nijianmo.github.io/amazon/index.html)

---

##  Reproducibility

- All configurations stored in: `config.yaml`
- Random seed fixed at: `42`
- All modules can be run independently and tested standalone

---

##  License

This project is licensed under the **MIT License © 2025 Shirmohammad Tavangari**

---


