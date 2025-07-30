from collections.abc import Iterable
from collections import Counter
from functools import lru_cache
import math
import random

class SmoothedNGramLanguageModel(object):
    # Initialize model with n-gram size, smoothing factor k, and vocab threshold.
    def __init__(self, n: int, k: float, threshold: int) -> None:
        self.n = n
        self.k = k
        self.threshold = threshold
        self.vocabulary = set()
        self.ngram_counts = {}
        self.context_counts = {}

    # Train model on sentences, build vocab and count n-grams.
    def train(self, training_sentences: list[list[str]]) -> None:
        self.vocabulary = self._build_vocabulary(training_sentences)
        for sentence in training_sentences:
            sentence = self._pad_tokens(sentence, self.n)
            self._update_counts(sentence)
        print(f"Number of n-grams: {len(self.ngram_counts)}")
        print(f"Number of contexts: {len(self.context_counts)}")

    # Add start and end tokens to sentence for n-gram processing.
    @staticmethod
    def _pad_tokens(tokens: list[str], n: int) -> list[str]:
        return ['<s>'] * (n - 1) + tokens + ['</s>']

    # Generate n-grams from tokens as tuples.
    @staticmethod
    def _generate_n_grams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i + n])

    # Build vocab from sentences, keep words with count >= threshold, add special tokens.
    def _build_vocabulary(self, training_sentences: list[list[str]]) -> set[str]:
        word_counts = Counter(word for sentence in training_sentences for word in sentence)
        vocab = {word for word, count in word_counts.items() if count >= self.threshold}
        vocab.add('<UNK>')
        vocab.add('<s>')
        vocab.add('</s>')
        return vocab

    # Update n-gram and context counts, replace unknown words with <UNK>.
    def _update_counts(self, sentence: list[str]) -> None:
        sentence = [word if word in self.vocabulary else '<UNK>' for word in sentence]
        for ngram in self._generate_n_grams(sentence, self.n):
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
            context = ngram[:-1]
            self.context_counts[context] = self.context_counts.get(context, 0) + 1

    # Calculate probability of sentences using log-probabilities with smoothing.
    def get_probability(self, sentences: list[list[str]]) -> float:
        log_prob = 0.0
        for sentence in sentences:
            sentence = self._pad_tokens([word if word in self.vocabulary else '<UNK>' for word in sentence], self.n)
            for ngram in self._generate_n_grams(sentence, self.n):
                context = ngram[:-1]
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(context, 0) or 1
                prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocabulary))
                log_prob += math.log(prob if prob > 0 else 1e-10)
        return math.exp(log_prob)

    # Compute perplexity of sentences, use log-prob for stability.
    def get_perplexity(self, sentences: list[list[str]]) -> float:
        log_prob = 0.0
        N = 0
        for sentence in sentences:
            padded_sentence = self._pad_tokens([word if word in self.vocabulary else '<UNK>' for word in sentence], self.n)
            for ngram in self._generate_n_grams(padded_sentence, self.n):
                context = ngram[:-1]
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(context, 0) or 1
                prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocabulary))
                log_prob += math.log(prob if prob > 0 else 1e-10)
            N += len(sentence)  # only count original words (no padding)
        return math.exp(-log_prob / N if N > 0 else float('inf'))

    # Generate random sentence using n-gram probs, start with history if given.
    def sample(self, random_seed: int, history: list[str] = []) -> Iterable[str]:
        random.seed(random_seed)
        sentence = history if history else ['<s>'] * (self.n - 1)
        max_length = 30
        min_length = 5
        while len(sentence) - (self.n - 1) < max_length and (sentence[-1] != '</s>' or len(sentence) - (self.n - 1) < min_length):
            context = tuple(sentence[-(self.n - 1):])
            next_words = {}
            for ngram, count in self.ngram_counts.items():
                if ngram[:-1] == context:
                    next_words[ngram[-1]] = (count + self.k) / (self.context_counts[context] + self.k * len(self.vocabulary))
            if not next_words:
                sentence.append('</s>')
                break
            words = [w for w in next_words.keys() if w != '<UNK>']
            if not words:
                sentence.append('</s>')
                break
            probs = [next_words[w] for w in words]
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            next_word = random.choices(words, weights=probs, k=1)[0]
            sentence.append(next_word)
        result = [word for word in sentence[(self.n - 1):] if word != '</s>']
        return result

    # Cache and return probability of n-gram with smoothing.
    @lru_cache(maxsize=100000)
    def get_ngram_prob(self, ngram: tuple[str, ...]) -> float:
        ctx, w = ngram[:-1], ngram[-1]
        cnt = self.ngram_counts.get(ngram, 0)
        cctx = self.context_counts.get(ctx, 0) or 1
        V = len(self.vocabulary)
        return (cnt + self.k) / (cctx + self.k * V)