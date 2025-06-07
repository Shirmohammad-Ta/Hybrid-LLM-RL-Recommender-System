
import pandas as pd
import os

def binarize_ratings(input_path, output_path, threshold=4):
    """
    Convert ratings to binary format: 1 if rating >= threshold, else 0.
    """
    df = pd.read_csv(input_path)
    if 'rating' not in df.columns:
        raise ValueError("Dataset must contain a 'rating' column.")

    df['binary_rating'] = (df['rating'] >= threshold).astype(int)
    df.to_csv(output_path, index=False)
    print(f"Saved binary data to {output_path}")

# Example usage
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    binarize_ratings("data/movielens_sample.csv", "data/processed/movielens_binary.csv")
    binarize_ratings("data/amazon_books_sample.csv", "data/processed/amazon_binary.csv")
