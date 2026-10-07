"""
CALLNGRAM N-Gram Language Models
Implements Unigram, Bigram, and Trigram language models with:
- Maximum Likelihood Estimation (MLE)
- Laplace (Add-1) smoothing
- Probability estimation
- Perplexity computation
- Frequency distributions
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


class NGramLanguageModel:
    """
    Unified N-gram Language Model supporting n=1 (unigram), n=2 (bigram), n=3 (trigram).
    Supports both unsmoothed MLE and Laplace (Add-1) smoothed probability estimation,
    along with transcript perplexity calculation.
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
        # Include </s> in effective target vocabulary size (|V| + 1)
        self.vocab_size = max(len(self.vocabulary) + 1, 1)
        self.total_tokens = total_toks
        return self

    def probability(self, word: str, context: Optional[Any] = None, smoothed: bool = True) -> float:
        """
        Calculates N-gram probability P(word | context).

        If smoothed=True (default), computes Laplace (Add-1) smoothed probability:
          - Unigram: P(w) = (count(w) + 1) / (N + V)
          - Bigram/Trigram: P(w | ctx) = (count(ctx, w) + 1) / (count(ctx) + V)
        where V is the fixed effective vocabulary size.

        If smoothed=False, computes Maximum Likelihood Estimation (MLE):
          - Unigram: P(w) = count(w) / N
          - Bigram/Trigram: P(w | ctx) = count(ctx, w) / count(ctx) if count(ctx) > 0 else 0.0
        """
        v = self.vocab_size

        if not smoothed:
            # Unsmoothed Maximum Likelihood Estimation
            if self.n == 1:
                if self.total_tokens == 0:
                    return 0.0
                return self.ngram_counts.get(word, 0) / float(self.total_tokens)
            elif self.n == 2:
                ctx = context
                c_ctx = self.context_counts.get(ctx, 0)
                if c_ctx == 0:
                    return 0.0
                return self.ngram_counts.get((ctx, word), 0) / float(c_ctx)
            elif self.n == 3:
                ctx = context
                c_ctx = self.context_counts.get(ctx, 0)
                if c_ctx == 0:
                    return 0.0
                return self.ngram_counts.get((ctx[0], ctx[1], word), 0) / float(c_ctx)

        # Laplace (Add-1) Smoothing
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

    def mle_probability(self, word: str, context: Optional[Any] = None) -> float:
        """Convenience method for unsmoothed Maximum Likelihood Estimation."""
        return self.probability(word, context=context, smoothed=False)

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
                prob = self.probability(tok, smoothed=True)
                log_prob_sum += math.log(prob)
                evaluated_count += 1

        elif self.n == 2:
            for i in range(1, len(sentence_tokens)):
                ctx = sentence_tokens[i - 1]
                target = sentence_tokens[i]
                prob = self.probability(target, context=ctx, smoothed=True)
                log_prob_sum += math.log(prob)
                evaluated_count += 1

        elif self.n == 3:
            for i in range(2, len(sentence_tokens)):
                ctx = (sentence_tokens[i - 2], sentence_tokens[i - 1])
                target = sentence_tokens[i]
                prob = self.probability(target, context=ctx, smoothed=True)
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

            # Calculate Laplace smoothed probability
            if self.n == 1:
                prob = self.probability(ngram, smoothed=True)
                phrase = ngram
            elif self.n == 2:
                prob = self.probability(ngram[1], context=ngram[0], smoothed=True)
                phrase = f"{ngram[0]} {ngram[1]}"
            elif self.n == 3:
                prob = self.probability(ngram[2], context=(ngram[0], ngram[1]), smoothed=True)
                phrase = f"{ngram[0]} {ngram[1]} {ngram[2]}"

            items.append((phrase, count, prob))

        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]
