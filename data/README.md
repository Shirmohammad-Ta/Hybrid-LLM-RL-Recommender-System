# Data Folder

This folder contains sample datasets and preprocessing scripts used in our hybrid recommender system project.

## Contents

- `movielens_sample.csv`: A small subset of the MovieLens dataset containing user ratings.
- `amazon_books_sample.csv`: A sample dataset of Amazon book reviews with user ratings and reviews.
- `preprocess.py`: A Python script that converts ratings into binary format (1 for relevant items, 0 otherwise).

## Instructions

To run preprocessing and generate binary datasets:

```bash
python preprocess.py
