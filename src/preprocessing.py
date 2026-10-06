"""
CALLNGRAM Preprocessing Module
Handles text cleaning, spoken marker preservation, tokenization, sentence segmentation,
and dataset auditing for AppTek Call-Center Dialogues and Bitext Customer Support.
"""

import re
import string
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import nltk

def ensure_nltk_resources():
    """Ensures required NLTK tokenizers are available with graceful fallbacks."""
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except (LookupError, Exception):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


ensure_nltk_resources()

# Spoken conversational markers to explicitly preserve
SPOKEN_MARKERS = {
    "uh", "um", "hmm", "hm", "okay", "ok", "yeah", "yep", "mhm", "ah", "oh", "woah"
}


def clean_text_for_tokens(text: str) -> str:
    """
    Cleans transcript text while strictly preserving conversational speech markers.
    - Strips bracket artifacts like (uh), (um,), [inaudible] -> converts to 'uh', 'um'
    - Strips tildes, backticks, unusual punctuation
    - Normalizes multiple spaces and lowercases
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    # Normalize speech markers in parentheses like (uh) -> uh, (um,) -> um
    text = re.sub(r'\(\s*(uh|um|hmm|hm|ah|oh)\s*,?\s*\)', r' \1 ', text, flags=re.IGNORECASE)
    # Remove remaining general parenthetical annotations like (inaudible) or [laughter]
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)
    # Remove conversational transcription markers like word interruptions: 'cab~' -> 'cab'
    text = re.sub(r'~+', ' ', text)
    # Replace non-standard dashes, quotes, accents
    text = text.replace("´", "'").replace("`", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def segment_sentences(text: str) -> List[str]:
    """Segment clean text into sentences with robust fallback."""
    if not text:
        return []
    cleaned = clean_text_for_tokens(text)
    if not cleaned:
        return []
    try:
        sentences = sent_tokenize(cleaned)
    except Exception:
        # Robust regex-based fallback if punkt / punkt_tab is unavailable in offline environment
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]
    return [s.strip() for s in sentences if s.strip()]


def tokenize_sentence(sentence: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes a sentence into words, filtering out standalone punctuations
    while keeping spoken markers and valid words.
    """
    if not sentence:
        return []
    if lowercase:
        sentence = sentence.lower()

    try:
        raw_tokens = word_tokenize(sentence)
    except Exception:
        # Robust fallback using regex if word_tokenize fails
        raw_tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", sentence)

    clean_tokens = []
    punct_set = set(string.punctuation) | {"''", "``", "--", "...", "’", "”", "“"}

    for tok in raw_tokens:
        tok_clean = tok.strip()
        # Keep word if not purely punctuation
        if tok_clean and tok_clean not in punct_set:
            # Strip lingering quotes/commas from edges
            tok_stripped = tok_clean.strip(string.punctuation)
            if tok_stripped:
                clean_tokens.append(tok_stripped)
            elif tok_clean in SPOKEN_MARKERS:
                clean_tokens.append(tok_clean)

    return clean_tokens


def tokenize_transcript(text: str, add_boundary_tokens: bool = False) -> Tuple[List[List[str]], List[str]]:
    """
    Preprocesses a transcript into:
    1. List of sentence token lists (optionally with <s> and </s>)
    2. Flat list of all words
    """
    sentences = segment_sentences(text)
    sent_tokens_list = []
    all_flat_tokens = []

    for s in sentences:
        toks = tokenize_sentence(s, lowercase=True)
        if not toks:
            continue
        if add_boundary_tokens:
            sent_with_bounds = ["<s>"] + toks + ["</s>"]
            sent_tokens_list.append(sent_with_bounds)
        else:
            sent_tokens_list.append(toks)
        all_flat_tokens.extend(toks)

    return sent_tokens_list, all_flat_tokens


def compute_transcript_statistics(transcripts: List[str]) -> Dict[str, Any]:
    """Computes detailed statistical summary for a corpus of transcripts."""
    lengths = []
    sentence_counts = []
    vocab_set = set()
    total_tokens = 0

    for t in transcripts:
        _, flat_toks = tokenize_transcript(t)
        tok_count = len(flat_toks)
        lengths.append(tok_count)
        sents = segment_sentences(t)
        sentence_counts.append(len(sents))
        total_tokens += tok_count
        vocab_set.update(flat_toks)

    lengths_arr = np.array(lengths) if lengths else np.array([0])
    sent_arr = np.array(sentence_counts) if sentence_counts else np.array([0])

    return {
        "total_transcripts": len(transcripts),
        "total_words": int(total_tokens),
        "vocabulary_size": len(vocab_set),
        "avg_words_per_transcript": float(np.mean(lengths_arr)),
        "median_words_per_transcript": float(np.median(lengths_arr)),
        "min_words_per_transcript": int(np.min(lengths_arr)),
        "max_words_per_transcript": int(np.max(lengths_arr)),
        "total_sentences": int(np.sum(sent_arr)),
        "avg_sentences_per_transcript": float(np.mean(sent_arr)),
    }
