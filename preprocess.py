import re
import string

import numpy as np
from collections import Counter


class Word2VecData:
    def __init__(self, min_freq=5, subsample_t=1e-5, table_size=10_000_000):
        self.min_freq = min_freq
        self.t = subsample_t
        self.table_size = table_size

        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.unigram_table = None

    def process(self, text):
        """Full preprocessing pipeline: tokenize -> filter -> subsample -> build negative sampling table."""
        print("1. Tokenizing and filtering rare words...")
        clean_text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
        words = clean_text.lower().split()

        raw_counts = Counter(words)

        # Drop words below min_freq entirely
        processed_words = [w for w in words if raw_counts[w] >= self.min_freq]

        word_counts = Counter(processed_words)
        total_words = len(processed_words)

        print("2. Building vocabulary...")
        unique_words = list(word_counts.keys())
        self.word_to_id = {w: i for i, w in enumerate(unique_words)}
        self.id_to_word = {i: w for i, w in enumerate(unique_words)}
        self.vocab_size = len(self.word_to_id)

        corpus_ids = [self.word_to_id[w] for w in processed_words]

        print("3. Subsampling frequent words...")
        freq_fractions = np.array([word_counts[self.id_to_word[i]] / total_words for i in range(self.vocab_size)])

        # P(keep) = sqrt(t / f(w)) — rare words kept more often, common words dropped more often
        p_keep = np.sqrt(self.t / freq_fractions)
        p_keep = np.clip(p_keep, 0, 1)

        random_rolls = np.random.random(len(corpus_ids))
        keep_thresholds = p_keep[corpus_ids]
        keep_mask = random_rolls < keep_thresholds
        self.train_corpus = np.array(corpus_ids)[keep_mask].tolist()

        print("4. Generating Negative Sampling Table...")
        self._build_unigram_table(word_counts)

        print(f"Done! Vocab size: {self.vocab_size} | Training words: {len(self.train_corpus)}")
        return self.train_corpus

    def _build_unigram_table(self, word_counts):
        """Builds the unigram table weighted by f(w)^0.75, which flattens the frequency distribution
        so rare words get sampled more often than raw counts would allow."""
        pow_freqs = np.array([word_counts[self.id_to_word[i]] ** 0.75 for i in range(self.vocab_size)])
        probs = pow_freqs / np.sum(pow_freqs)

        # Each word gets a slice of the table proportional to its smoothed frequency
        # np.repeat([0,1], [2,3]) -> [0, 0, 1, 1, 1]
        counts = np.round(probs * self.table_size).astype(int)
        self.unigram_table = np.repeat(np.arange(self.vocab_size), counts)
        np.random.shuffle(self.unigram_table)

    def get_negative_samples(self, num_samples):
        """Draws negative samples from the precomputed table in O(1)."""
        indices = np.random.randint(0, len(self.unigram_table), size=num_samples)
        return self.unigram_table[indices]