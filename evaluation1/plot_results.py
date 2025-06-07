import matplotlib.pyplot as plt

def plot_bar(metrics, models, metric_name):
    values = [metrics[m] for m in models]
    plt.bar(models, values)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric_name.lower()}_bar.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    precision_scores = {
        "MF": 0.61,
        "DeepMF": 0.67,
        "DDPG": 0.71,
        "Proposed": 0.83
    }
    plot_bar(precision_scores, list(precision_scores.keys()), "Precision@10")
