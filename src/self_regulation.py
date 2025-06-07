
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SelfRegulator:
    def __init__(self, diversity_threshold=0.5, min_length=10):
        self.diversity_threshold = diversity_threshold
        self.min_length = min_length
        self.vectorizer = TfidfVectorizer()

    def score_sample(self, text):
        """Returns a score (0 to 1) based on length and uniqueness"""
        length_score = min(len(text) / 100, 1.0)
        return length_score

    def filter_batch(self, samples):
        """
        Filters out low-quality or highly redundant synthetic samples.
        Returns a list of high-quality samples.
        """
        filtered = [s for s in samples if len(s) >= self.min_length]
        if len(filtered) < 2:
            return filtered

        tfidf_matrix = self.vectorizer.fit_transform(filtered)
        sim_matrix = cosine_similarity(tfidf_matrix)

        diverse_samples = []
        for i, s in enumerate(filtered):
            similarity_to_others = np.mean(np.delete(sim_matrix[i], i))
            if similarity_to_others < self.diversity_threshold:
                diverse_samples.append(s)

        return diverse_samples
