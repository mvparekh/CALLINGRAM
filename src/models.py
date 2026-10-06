"""
CALLNGRAM N-Gram Language Models
Implements Unigram, Bigram, and Trigram language models with:
- Frequency distributions
- Laplace (Add-1) smoothing
- Probability estimation
- Perplexity computation
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


class NGramLanguageModel:
    """
    Unified N-gram Language Model supporting n=1 (unigram), n=2 (bigram), n=3 (trigram).
    Implements Laplace / Add-1 smoothing and perplexity calculation.
    """

    def __init__(self, n: int = 1):
        if n not in (1, 2, 3):
            raise ValueError(f"Order n must be 1, 2, or 3, got {n}")
        self.n = n
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocabulary: set = set()
        self.total_tokens: int = 0
        self.vocab_size: int = 0

    def fit(self, tokenized_sentences: List[List[str]], vocab_override: Optional[set] = None):
        """
        Trains the N-gram language model over tokenized sentences.
        Each sentence should contain boundary tokens appropriate for the order.
        """
        self.ngram_counts.clear()
        self.context_counts.clear()
        vocab = set()
        total_toks = 0

        for sent in tokenized_sentences:
            if not sent:
                continue

            # Update vocabulary from non-boundary tokens
            for tok in sent:
                if tok not in ("<s>", "</s>"):
                    vocab.add(tok)
                total_toks += 1

            if self.n == 1:
                for tok in sent:
                    self.ngram_counts[tok] += 1
            elif self.n == 2:
                # Bigrams: (w_{i-1}, w_i)
                for i in range(1, len(sent)):
                    ctx = sent[i - 1]
                    ngram = (sent[i - 1], sent[i])
                    self.context_counts[ctx] += 1
                    self.ngram_counts[ngram] += 1
            elif self.n == 3:
                # Trigrams: ((w_{i-2}, w_{i-1}), w_i)
                for i in range(2, len(sent)):
                    ctx = (sent[i - 2], sent[i - 1])
                    ngram = (sent[i - 2], sent[i - 1], sent[i])
                    self.context_counts[ctx] += 1
                    self.ngram_counts[ngram] += 1

        self.vocabulary = vocab_override if vocab_override is not None else vocab
        # Include </s> in effective target vocabulary size
        self.vocab_size = max(len(self.vocabulary) + 1, 1)
        self.total_tokens = total_toks
        return self

    def probability(self, word: str, context: Optional[Any] = None) -> float:
        """
        Calculates Laplace (Add-1) smoothed probability: P(word | context).
        - Unigram: P(w) = (count(w) + 1) / (total_tokens + V)
        - Bigram/Trigram: P(w | ctx) = (count(ctx, w) + 1) / (count(ctx) + V)
        where V is the vocabulary size.
        """
        v = self.vocab_size

        if self.n == 1:
            c_w = self.ngram_counts.get(word, 0)
            return (c_w + 1.0) / (self.total_tokens + v)

        elif self.n == 2:
            ctx = context
            ngram = (ctx, word)
            c_ngram = self.ngram_counts.get(ngram, 0)
            c_ctx = self.context_counts.get(ctx, 0)
            return (c_ngram + 1.0) / (c_ctx + v)

        elif self.n == 3:
            ctx = context  # tuple of (w_{i-2}, w_{i-1})
            ngram = (ctx[0], ctx[1], word)
            c_ngram = self.ngram_counts.get(ngram, 0)
            c_ctx = self.context_counts.get(ctx, 0)
            return (c_ngram + 1.0) / (c_ctx + v)

        return 1.0 / v

    def score_sentence(self, sentence_tokens: List[str]) -> Tuple[float, int]:
        """
        Calculates the log probability sum and evaluated token count for a sentence.
        Returns: (sum_log_prob, evaluated_token_count)
        """
        if not sentence_tokens:
            return 0.0, 0

        log_prob_sum = 0.0
        evaluated_count = 0

        if self.n == 1:
            for tok in sentence_tokens:
                prob = self.probability(tok)
                log_prob_sum += math.log(prob)
                evaluated_count += 1

        elif self.n == 2:
            for i in range(1, len(sentence_tokens)):
                ctx = sentence_tokens[i - 1]
                target = sentence_tokens[i]
                prob = self.probability(target, context=ctx)
                log_prob_sum += math.log(prob)
                evaluated_count += 1

        elif self.n == 3:
            for i in range(2, len(sentence_tokens)):
                ctx = (sentence_tokens[i - 2], sentence_tokens[i - 1])
                target = sentence_tokens[i]
                prob = self.probability(target, context=ctx)
                log_prob_sum += math.log(prob)
                evaluated_count += 1

        return log_prob_sum, evaluated_count

    def perplexity(self, tokenized_sentences: List[List[str]]) -> float:
        """
        Calculates transcript perplexity:
        PP = exp( -(1 / N) * sum(log P(w_i | context)) )
        where N is total evaluated tokens across the transcript.
        """
        total_log_prob = 0.0
        total_eval_tokens = 0

        for sent in tokenized_sentences:
            l_prob, count = self.score_sentence(sent)
            total_log_prob += l_prob
            total_eval_tokens += count

        if total_eval_tokens == 0:
            return float("inf")

        cross_entropy = - (total_log_prob / total_eval_tokens)
        try:
            pp = math.exp(cross_entropy)
            return float(pp)
        except OverflowError:
            return float("inf")

    def get_top_ngrams(self, top_k: int = 50, exclude_boundaries: bool = True) -> List[Tuple[str, int, float]]:
        """
        Returns top N-grams sorted by count with Add-1 smoothed probability.
        Format: List of ("phrase", count, probability)
        """
        items = []
        for ngram, count in self.ngram_counts.items():
            if exclude_boundaries:
                if isinstance(ngram, str) and ngram in ("<s>", "</s>"):
                    continue
                if isinstance(ngram, tuple) and any(tok in ("<s>", "</s>") for tok in ngram):
                    continue

            # Calculate probability
            if self.n == 1:
                prob = self.probability(ngram)
                phrase = ngram
            elif self.n == 2:
                prob = self.probability(ngram[1], context=ngram[0])
                phrase = f"{ngram[0]} {ngram[1]}"
            elif self.n == 3:
                prob = self.probability(ngram[2], context=(ngram[0], ngram[1]))
                phrase = f"{ngram[0]} {ngram[1]} {ngram[2]}"

            items.append((phrase, count, prob))

        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]
