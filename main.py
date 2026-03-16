import numpy as np
import matplotlib.pyplot as plt
from model import Word2VecModel
from preprocess import Word2VecData


def main():
    with open("enwik8", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataset = Word2VecData(min_freq=5)
    corpus = dataset.process(raw_text)

    model = Word2VecModel(dataset.vocab_size, embedding_dim=128)

    losses = model.train(
        dataset=dataset,
        corpus=corpus,
        epochs=2,
        initial_lr=0.025,
        window_size=2,
        num_negatives=5,
    )

    # Sum W_in + W_out — common alternative to using W_in alone
    final_embeddings = model.W_in + model.W_out
    print("\nTraining complete. Embedding matrix shape:", final_embeddings.shape)

    # Plot losses
    plot_loss(losses)

    # Prove the king - man + woman = queen analogy
    analogy(final_embeddings, dataset)

    test_words = ["king", "queen", "man", "woman", "apple", "orange", "dog", "cat"]
    plot_embeddings_pca(test_words, final_embeddings, dataset)

    print("\nMost similar words to 'apple':")
    get_similar_words("apple", final_embeddings, dataset)


def plot_loss(losses):
    steps = np.arange(len(losses))

    # Smooth the raw losses with a rolling average so the plot is readable
    window = 200
    smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
    smoothed_steps = steps[:len(smoothed)]

    # Fit a degree-4 polynomial to the smoothed curve
    coeffs = np.polyfit(smoothed_steps, smoothed, deg=4)
    fit_line = np.polyval(coeffs, smoothed_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, alpha=0.2, color='steelblue', label='raw loss')
    plt.plot(smoothed_steps, smoothed, color='steelblue', label='smoothed')
    plt.plot(smoothed_steps, fit_line, color='red', linewidth=2, label='best fit')
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig("loss_curve.png")
    plt.show()

def analogy(embeddings, dataset):
    """Classic king - man + woman analogy test."""
    try:
        king = embeddings[dataset.word_to_id["king"]]
        man = embeddings[dataset.word_to_id["man"]]
        woman = embeddings[dataset.word_to_id["woman"]]
    except KeyError as e:
        print(f"Analogy skipped — word not in vocab: {e}")
        return

    result_vec = king - man + woman
    print("\nking - man + woman → nearest neighbors:")
    get_similar_words_from_vector(result_vec, embeddings, dataset)


def get_similar_words(target_word, embeddings, dataset, top_k=5):
    """Cosine similarity search for a word in the vocabulary."""
    if target_word not in dataset.word_to_id:
        print(f"'{target_word}' not in vocabulary.")
        return

    target_vec = embeddings[dataset.word_to_id[target_word]]
    get_similar_words_from_vector(target_vec, embeddings, dataset, top_k, skip_top1=True)


def get_similar_words_from_vector(target_vec, embeddings, dataset, top_k=5, skip_top1=False):
    """Cosine similarity search from a raw vector (used by both get_similar_words and analogy)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    target_norm = np.linalg.norm(target_vec)
    normalized_target = target_vec / (target_norm + 1e-10)

    similarities = np.dot(normalized, normalized_target)

    # skip index 0 if the target itself is in the matrix (e.g. get_similar_words)
    start = 1 if skip_top1 else 0
    top_indices = np.argsort(similarities)[::-1][start: start + top_k]

    for i in top_indices:
        print(f"  {dataset.id_to_word[i]}: {similarities[i]:.4f}")


def plot_embeddings_pca(words, embeddings, dataset):
    """PCA from scratch using NumPy — projects word vectors down to 2D for visualization."""
    indices = [dataset.word_to_id[w] for w in words if w in dataset.word_to_id]
    valid_words = [dataset.id_to_word[i] for i in indices]
    X = embeddings[indices]

    if len(X) == 0:
        print("No words found in vocabulary for PCA plot.")
        return

    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order, so reverse and take top 2
    top2 = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, :2]
    X_2d = X_centered @ top2

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="steelblue", edgecolors="k", alpha=0.7)
    for i, word in enumerate(valid_words):
        plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]), textcoords="offset points",
                     xytext=(5, 2), ha="right", va="bottom")
    plt.title("Word2Vec — PCA projection")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig("pca.png")
    plt.show()


if __name__ == "__main__":
    main()