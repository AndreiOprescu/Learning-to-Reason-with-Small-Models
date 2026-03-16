import numpy as np


def sigmoid(x):
    # Clipped to [-10, 10] to avoid overflow in exp
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class Word2VecModel:
    def __init__(self, vocab_size, embedding_dim=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Uniform init scaled by embedding size, same as original word2vec
        bound = 0.5 / self.embedding_dim
        self.W_in = np.random.uniform(-bound, bound, size=(vocab_size, embedding_dim))
        self.W_out = np.random.uniform(-bound, bound, size=(vocab_size, embedding_dim))

    def train_step_batched(self, center_ids, context_ids, negative_ids, lr):
        """
        Skip-gram negative sampling forward + backward pass for a single batch.

        center_ids:   (B,)
        context_ids:  (B,)
        negative_ids: (B, K)
        """

        # --- FORWARD ---
        v_c = self.W_in[center_ids]       # (B, D)
        u_pos = self.W_out[context_ids]   # (B, D)
        u_neg = self.W_out[negative_ids]  # (B, K, D)

        score_pos = np.sum(v_c * u_pos, axis=1)                          # (B,)
        score_neg = np.sum(v_c[:, np.newaxis, :] * u_neg, axis=2)        # (B, K)

        prob_pos = sigmoid(score_pos)
        prob_neg = sigmoid(score_neg)

        # Negative sampling loss: log σ(v·u_pos) + Σ log σ(-v·u_neg)
        loss = np.mean(-np.log(prob_pos + 1e-10) - np.sum(np.log(1 - prob_neg + 1e-10), axis=1))

        # --- BACKWARD ---
        # Gradient of loss w.r.t. scores: σ(score) - label
        error_pos = prob_pos - 1.0   # (B,)
        error_neg = prob_neg         # (B, K)  — label is 0, so σ - 0 = σ

        grad_u_pos = error_pos[:, np.newaxis] * v_c                                    # (B, D)
        grad_u_neg = error_neg[:, :, np.newaxis] * v_c[:, np.newaxis, :]              # (B, K, D)
        grad_v_c = (error_pos[:, np.newaxis] * u_pos) + \
                   np.sum(error_neg[:, :, np.newaxis] * u_neg, axis=1)                # (B, D)

        grad_v_c   = np.clip(grad_v_c,   -5, 5)
        grad_u_pos = np.clip(grad_u_pos, -5, 5)
        grad_u_neg = np.clip(grad_u_neg, -5, 5)

        # --- UPDATE ---
        # np.add.at handles duplicate IDs in a batch correctly (no overwrite)
        np.add.at(self.W_out, context_ids, -lr * grad_u_pos)
        np.add.at(self.W_out, negative_ids.flatten(), -lr * grad_u_neg.reshape(-1, self.embedding_dim))
        np.add.at(self.W_in, center_ids, -lr * grad_v_c)

        return loss

    def generate_training_pairs(self, corpus, window_size=5):
        """Yields (center, context) pairs using a dynamic window — closer words
        are sampled more often because the window shrinks randomly each step."""
        corpus_length = len(corpus)

        for i, center_word_id in enumerate(corpus):
            dynamic_window = np.random.randint(1, window_size + 1)
            start = max(0, i - dynamic_window)
            end = min(corpus_length, i + dynamic_window + 1)

            for j in range(start, end):
                if i != j:
                    yield center_word_id, corpus[j]

    def train(self, dataset, corpus, epochs=5, initial_lr=0.025, window_size=5, num_negatives=5, batch_size=128):
        print(f"Starting training: {epochs} epochs, batch_size={batch_size}")

        losses = []

        # Approximate total steps for linear LR decay
        estimated_pairs = len(corpus) * 2 * window_size
        total_steps = (estimated_pairs / batch_size) * epochs
        global_step = 0

        for epoch in range(epochs):
            pair_generator = self.generate_training_pairs(corpus, window_size)
            batch_centers, batch_contexts = [], []
            step = 0

            for center_id, context_id in pair_generator:
                batch_centers.append(center_id)
                batch_contexts.append(context_id)

                if len(batch_centers) == batch_size:
                    centers_arr   = np.array(batch_centers)
                    contexts_arr  = np.array(batch_contexts)
                    negatives_arr = dataset.get_negative_samples(batch_size * num_negatives)
                    negatives_arr = negatives_arr.reshape(batch_size, num_negatives)

                    lr = initial_lr * max(1.0 - global_step / total_steps, 0.0001)
                    loss = self.train_step_batched(centers_arr, contexts_arr, negatives_arr, lr)
                    losses.append(loss)

                    batch_centers, batch_contexts = [], []
                    global_step += 1
                    step += 1

                    if step % 1000 == 0:
                        print(f"Epoch {epoch + 1}/{epochs} | step {step} | loss {loss:.4f} | lr {lr:.6f}")

            # Handle the last partial batch
            if batch_centers:
                lr = initial_lr * max(1.0 - global_step / total_steps, 0.0001)
                centers_arr   = np.array(batch_centers)
                contexts_arr  = np.array(batch_contexts)
                negatives_arr = dataset.get_negative_samples(len(batch_centers) * num_negatives)
                negatives_arr = negatives_arr.reshape(len(batch_centers), num_negatives)
                self.train_step_batched(centers_arr, contexts_arr, negatives_arr, lr)
                global_step += 1

            print(f"Epoch {epoch + 1} done. lr={lr:.6f}")

        return losses