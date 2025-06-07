# 🔮 Hybrid LLM-RL Recommender System

This project implements a novel hybrid recommender system that combines **Large Language Models (LLMs)** with **Reinforcement Learning (RL)** and a **Self-Regulation Mechanism** to improve recommendation quality, especially in sparse-data environments.

---

## 🚀 Key Features

- 🤖 Synthetic data generation using GPT-based LLMs
- 🧠 Reinforcement Learning with PPO for feedback-driven optimization
- 🔄 Self-regulation module to select high-quality, diverse training samples
- 📈 Evaluation on MovieLens and Amazon Books datasets
- ✅ Designed for cold-start and low-data scenarios

---

## 📁 Project Structure

<pre> <code> ``` Hybrid-LLM-RL-Recommender-System/ │ ├── data/ │ ├── movielens_sample.csv │ ├── amazon_books_sample.csv │ ├── preprocess.py │ └── processed/ │ ├── src/ │ ├── llm_generator.py │ ├── rl_agent.py │ ├── self_regulation.py │ ├── train.py │ └── utils.py │ ├── evaluation/ │ ├── evaluate.py │ ├── plot_results.py │ ├── RESULTS.md │ └── README.md │ ├── config.yaml └── README.md ← (this file) ``` </code> </pre>
