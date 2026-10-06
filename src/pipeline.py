"""
CALLNGRAM Full End-to-End Pipeline
Executes preprocessing, train/test splitting, N-gram modeling, Laplace smoothing,
perplexity evaluation, phrase frequency & probability reporting, and customer issue intelligence.
"""

import os
import sys
import json
import math
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Support execution from project root or inside CALLNGRAM
ROOT_DIR = Path(__file__).resolve().parent.parent
if (ROOT_DIR / "src").exists():
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    clean_text_for_tokens,
    segment_sentences,
    tokenize_sentence,
    tokenize_transcript,
    compute_transcript_statistics
)
from src.models import NGramLanguageModel


def run_full_pipeline(base_dir: Path = None):
    if base_dir is None:
        base_dir = ROOT_DIR

    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw_transcripts"
    ann_dir = data_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      CALLNGRAM PIPELINE EXECUTION START          ")
    print("==================================================")

    # ----------------------------------------------------
    # 1. LOAD & CLEAN APPTEK CORPUS (PRIMARY DATASET)
    # ----------------------------------------------------
    print("\n[STEP 1] Loading and auditing AppTek Call-Center Dialogues...")
    apptek_jsonl = raw_dir / "apptek" / "apptek_dialogues_consolidated.jsonl"
    if not apptek_jsonl.exists():
        raise FileNotFoundError(f"Consolidated AppTek file not found at: {apptek_jsonl}")

    apptek_records = []
    with open(apptek_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                apptek_records.append(json.loads(line))

    df_apptek = pd.DataFrame(apptek_records)
    total_apptek = len(df_apptek)
    missing_texts = df_apptek["text"].isna().sum()
    duplicate_texts = df_apptek["text"].duplicated().sum()
    unique_domains = sorted(df_apptek["domain"].dropna().unique().tolist())
    unique_accents = sorted(df_apptek["accent"].dropna().unique().tolist())
    customer_records = (df_apptek["role"].str.lower() == "customer").sum()
    agent_records = (df_apptek["role"].str.lower() == "agent").sum()

    print(f"  - Total AppTek records: {total_apptek}")
    print(f"  - Missing text records: {missing_texts}")
    print(f"  - Duplicate transcript texts: {duplicate_texts}")
    print(f"  - Service domains ({len(unique_domains)}): {unique_domains}")
    print(f"  - Accent groups ({len(unique_accents)}): {unique_accents}")
    print(f"  - Customer transcripts: {customer_records} | Agent transcripts: {agent_records}")

    # Clean AppTek texts and extract length metadata
    cleaned_texts = []
    token_counts = []
    sentence_counts = []

    for text in df_apptek["text"]:
        c_text = clean_text_for_tokens(text)
        cleaned_texts.append(c_text)
        sents = segment_sentences(c_text)
        sentence_counts.append(len(sents))
        # Count words
        words = []
        for s in sents:
            words.extend(tokenize_sentence(s))
        token_counts.append(len(words))

    df_apptek["clean_text"] = cleaned_texts
    df_apptek["word_count"] = token_counts
    df_apptek["sentence_count"] = sentence_counts

    avg_words = float(np.mean(token_counts))
    median_words = float(np.median(token_counts))
    print(f"  - Average transcript length: {avg_words:.1f} words (median: {median_words:.1f})")

    apptek_clean_path = ann_dir / "apptek_clean.csv"
    df_apptek.to_csv(apptek_clean_path, index=False)
    print(f"  [OK] Saved cleaned AppTek corpus: {apptek_clean_path}")

    # ----------------------------------------------------
    # 2. TRANSCRIPT-LEVEL TRAIN / TEST SPLIT (80% / 20%)
    # ----------------------------------------------------
    print("\n[STEP 2] Performing transcript-level train/test split (80/20, seed=42)...")
    train_df, test_df = train_test_split(df_apptek, test_size=0.20, random_state=42)
    print(f"  - Training transcripts: {len(train_df)} (80%)")
    print(f"  - Testing transcripts: {len(test_df)} (20%)")
    print("  - Note: Entire transcripts held out to prevent sentence-level data leakage.")

    # ----------------------------------------------------
    # 3. TOKENIZE CORPUS FOR N-GRAM MODELING
    # ----------------------------------------------------
    print("\n[STEP 3] Preparing training sentences with boundary tokens...")
    train_sents_1gram = []
    train_sents_2gram = []
    train_sents_3gram = []
    train_vocab = set()

    for text in train_df["clean_text"]:
        sents = segment_sentences(text)
        for s in sents:
            toks = tokenize_sentence(s, lowercase=True)
            if not toks:
                continue
            train_vocab.update(toks)
            train_sents_1gram.append(toks)
            train_sents_2gram.append(["<s>"] + toks + ["</s>"])
            train_sents_3gram.append(["<s>", "<s>"] + toks + ["</s>"])

    print(f"  - Total training sentences: {len(train_sents_1gram):,}")
    print(f"  - Training vocabulary size: {len(train_vocab):,} unique words")

    # ----------------------------------------------------
    # 4. FIT UNIGRAM, BIGRAM, TRIGRAM WITH LAPLACE SMOOTHING
    # ----------------------------------------------------
    print("\n[STEP 4] Fitting N-Gram Language Models with Laplace / Add-1 Smoothing...")
    model_unigram = NGramLanguageModel(n=1)
    model_unigram.fit(train_sents_1gram, vocab_override=train_vocab)

    model_bigram = NGramLanguageModel(n=2)
    model_bigram.fit(train_sents_2gram, vocab_override=train_vocab)

    model_trigram = NGramLanguageModel(n=3)
    model_trigram.fit(train_sents_3gram, vocab_override=train_vocab)

    print("  [OK] Unigram, Bigram, and Trigram models fitted.")

    # ----------------------------------------------------
    # 5. PERPLEXITY EVALUATION ON TEST TRANSCRIPTS
    # ----------------------------------------------------
    print("\n[STEP 5] Evaluating Perplexity on unseen test transcripts (N=350)...")
    test_pp_unigram = []
    test_pp_bigram = []
    test_pp_trigram = []

    for text in test_df["clean_text"]:
        sents = segment_sentences(text)
        t_sents_1 = []
        t_sents_2 = []
        t_sents_3 = []
        for s in sents:
            toks = tokenize_sentence(s, lowercase=True)
            if not toks:
                continue
            t_sents_1.append(toks)
            t_sents_2.append(["<s>"] + toks + ["</s>"])
            t_sents_3.append(["<s>", "<s>"] + toks + ["</s>"])

        pp1 = model_unigram.perplexity(t_sents_1)
        pp2 = model_bigram.perplexity(t_sents_2)
        pp3 = model_trigram.perplexity(t_sents_3)

        if math.isfinite(pp1):
            test_pp_unigram.append(pp1)
        if math.isfinite(pp2):
            test_pp_bigram.append(pp2)
        if math.isfinite(pp3):
            test_pp_trigram.append(pp3)

    perplexity_report_data = [
        {
            "model": "Unigram (Add-1)",
            "mean_perplexity": round(float(np.mean(test_pp_unigram)), 2),
            "median_perplexity": round(float(np.median(test_pp_unigram)), 2),
            "transcripts_evaluated": len(test_pp_unigram),
        },
        {
            "model": "Bigram (Add-1)",
            "mean_perplexity": round(float(np.mean(test_pp_bigram)), 2),
            "median_perplexity": round(float(np.median(test_pp_bigram)), 2),
            "transcripts_evaluated": len(test_pp_bigram),
        },
        {
            "model": "Trigram (Add-1)",
            "mean_perplexity": round(float(np.mean(test_pp_trigram)), 2),
            "median_perplexity": round(float(np.median(test_pp_trigram)), 2),
            "transcripts_evaluated": len(test_pp_trigram),
        },
    ]

    df_pp = pd.DataFrame(perplexity_report_data)
    pp_path = ann_dir / "perplexity_report.csv"
    df_pp.to_csv(pp_path, index=False)
    print("  [OK] Perplexity evaluation summary:")
    print(df_pp.to_string(index=False))

    # ----------------------------------------------------
    # 6. GENERATE FREQUENCY & PROBABILITY REPORTS
    # ----------------------------------------------------
    print("\n[STEP 6] Generating Frequency and Probability CSV reports...")
    top_1g = model_unigram.get_top_ngrams(top_k=500, exclude_boundaries=True)
    top_2g = model_bigram.get_top_ngrams(top_k=500, exclude_boundaries=True)
    top_3g = model_trigram.get_top_ngrams(top_k=500, exclude_boundaries=True)

    # Frequency reports (ngram, count)
    df_1g_freq = pd.DataFrame([{"ngram": p, "count": c} for p, c, _ in top_1g])
    df_2g_freq = pd.DataFrame([{"ngram": p, "count": c} for p, c, _ in top_2g])
    df_3g_freq = pd.DataFrame([{"ngram": p, "count": c} for p, c, _ in top_3g])

    df_1g_freq.to_csv(ann_dir / "apptek_1gram_frequency.csv", index=False)
    df_2g_freq.to_csv(ann_dir / "apptek_2gram_frequency.csv", index=False)
    df_3g_freq.to_csv(ann_dir / "apptek_3gram_frequency.csv", index=False)

    # Probability reports (ngram, count, probability)
    df_1g_prob = pd.DataFrame([{"ngram": p, "count": c, "probability": prob} for p, c, prob in top_1g])
    df_2g_prob = pd.DataFrame([{"ngram": p, "count": c, "probability": prob} for p, c, prob in top_2g])
    df_3g_prob = pd.DataFrame([{"ngram": p, "count": c, "probability": prob} for p, c, prob in top_3g])

    df_1g_prob.to_csv(ann_dir / "1gram_probability_report.csv", index=False)
    df_2g_prob.to_csv(ann_dir / "2gram_probability_report.csv", index=False)
    df_3g_prob.to_csv(ann_dir / "3gram_probability_report.csv", index=False)

    print("  [OK] Saved apptek_1gram_frequency.csv, apptek_2gram_frequency.csv, apptek_3gram_frequency.csv")
    print("  [OK] Saved 1gram_probability_report.csv, 2gram_probability_report.csv, 3gram_probability_report.csv")

    # ----------------------------------------------------
    # 7. BITEXT CUSTOMER ISSUE PHRASE INTELLIGENCE
    # ----------------------------------------------------
    print("\n[STEP 7] Loading and analyzing Bitext Customer Support Dataset (Secondary Corpus)...")
    bitext_csv = raw_dir / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    if not bitext_csv.exists():
        raise FileNotFoundError(f"Bitext dataset not found at: {bitext_csv}")

    df_bitext = pd.read_csv(bitext_csv)
    print(f"  - Total Bitext records: {len(df_bitext):,}")
    print(f"  - Categories ({df_bitext['category'].nunique()}): {sorted(df_bitext['category'].dropna().unique().tolist())}")
    print(f"  - Intents ({df_bitext['intent'].nunique()}): {sorted(df_bitext['intent'].dropna().unique().tolist())}")

    # Clean instructions
    clean_instructions = []
    b_tokens_count = []
    for inst in df_bitext["instruction"].fillna(""):
        c_i = clean_text_for_tokens(inst)
        clean_instructions.append(c_i)
        toks = tokenize_sentence(c_i)
        b_tokens_count.append(len(toks))

    df_bitext["clean_instruction"] = clean_instructions
    df_bitext["token_count"] = b_tokens_count
    df_bitext_clean = df_bitext[["instruction", "clean_instruction", "category", "intent", "token_count"]]
    df_bitext_clean.to_csv(ann_dir / "bitext_clean.csv", index=False)
    print(f"  [OK] Saved cleaned Bitext dataset: {ann_dir / 'bitext_clean.csv'}")

    # Generate Category-wise and Intent-wise N-gram phrase reports
    print("  - Generating category and intent-specific N-gram phrases...")
    issue_records = []

    def extract_ngrams_from_texts(texts: List[str], order: int, top_n: int = 10) -> List[Tuple[str, int]]:
        counts = {}
        for text in texts:
            toks = tokenize_sentence(text)
            if len(toks) < order:
                continue
            for i in range(len(toks) - order + 1):
                phrase = " ".join(toks[i:i + order])
                counts[phrase] = counts.get(phrase, 0) + 1
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

    # Category analysis
    for cat, group in df_bitext.groupby("category"):
        texts = group["clean_instruction"].tolist()
        top_u = extract_ngrams_from_texts(texts, 1, 10)
        top_b = extract_ngrams_from_texts(texts, 2, 10)
        top_t = extract_ngrams_from_texts(texts, 3, 10)

        for phrase, c in top_u:
            issue_records.append({
                "level": "category",
                "target": cat,
                "ngram_order": "unigram",
                "phrase": phrase,
                "count": c
            })
        for phrase, c in top_b:
            issue_records.append({
                "level": "category",
                "target": cat,
                "ngram_order": "bigram",
                "phrase": phrase,
                "count": c
            })
        for phrase, c in top_t:
            issue_records.append({
                "level": "category",
                "target": cat,
                "ngram_order": "trigram",
                "phrase": phrase,
                "count": c
            })

    # Intent analysis
    for intent, group in df_bitext.groupby("intent"):
        texts = group["clean_instruction"].tolist()
        top_u = extract_ngrams_from_texts(texts, 1, 10)
        top_b = extract_ngrams_from_texts(texts, 2, 10)
        top_t = extract_ngrams_from_texts(texts, 3, 10)

        for phrase, c in top_u:
            issue_records.append({
                "level": "intent",
                "target": intent,
                "ngram_order": "unigram",
                "phrase": phrase,
                "count": c
            })
        for phrase, c in top_b:
            issue_records.append({
                "level": "intent",
                "target": intent,
                "ngram_order": "bigram",
                "phrase": phrase,
                "count": c
            })
        for phrase, c in top_t:
            issue_records.append({
                "level": "intent",
                "target": intent,
                "ngram_order": "trigram",
                "phrase": phrase,
                "count": c
            })

    df_issue_report = pd.DataFrame(issue_records)
    issue_report_path = ann_dir / "bitext_issue_phrase_report.csv"
    df_issue_report.to_csv(issue_report_path, index=False)
    print(f"  [OK] Saved Bitext issue phrase report: {issue_report_path} ({len(df_issue_report)} rows)")

    # ----------------------------------------------------
    # 8. PROJECT SUMMARY METADATA JSON
    # ----------------------------------------------------
    stats_apptek = compute_transcript_statistics(df_apptek["clean_text"].tolist())
    summary = {
        "project_name": "CALLNGRAM",
        "description": "Customer Call Language Analytics via N-gram Language Modeling",
        "apptek": {
            "total_transcripts": total_apptek,
            "train_transcripts": len(train_df),
            "test_transcripts": len(test_df),
            "domains": unique_domains,
            "accents": unique_accents,
            "customer_records": int(customer_records),
            "agent_records": int(agent_records),
            "total_words": stats_apptek["total_words"],
            "vocabulary_size": stats_apptek["vocabulary_size"],
            "avg_words_per_transcript": round(stats_apptek["avg_words_per_transcript"], 2),
            "median_words_per_transcript": round(stats_apptek["median_words_per_transcript"], 2),
            "min_words_per_transcript": stats_apptek["min_words_per_transcript"],
            "max_words_per_transcript": stats_apptek["max_words_per_transcript"],
            "total_sentences": stats_apptek["total_sentences"],
        },
        "bitext": {
            "total_examples": len(df_bitext),
            "categories_count": int(df_bitext["category"].nunique()),
            "categories": sorted(df_bitext["category"].dropna().unique().tolist()),
            "intents_count": int(df_bitext["intent"].nunique()),
            "intents": sorted(df_bitext["intent"].dropna().unique().tolist()),
        },
        "models_evaluated": perplexity_report_data,
    }

    summary_path = ann_dir / "project_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # ----------------------------------------------------
    # 9. MIRROR KEY ARTIFACTS TO REPORTS/ AND GENERATE PDFS
    # ----------------------------------------------------
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for rep_file in ["perplexity_report.csv", "1gram_probability_report.csv", "2gram_probability_report.csv", "3gram_probability_report.csv", "bitext_issue_phrase_report.csv", "project_summary.json"]:
        src_f = ann_dir / rep_file
        if src_f.exists():
            shutil.copy2(src_f, reports_dir / rep_file)
    print("  [OK] Mirrored CSV & JSON reports to reports/")

    # Generate PDF documentation artifacts
    try:
        from src.generate_reports import main as generate_pdf_reports
        generate_pdf_reports()
    except Exception as e:
        print(f"  [WARN] PDF generation warning: {e}")

    print("\n==================================================")
    print("      CALLNGRAM PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("==================================================")
    return summary


if __name__ == "__main__":
    run_full_pipeline()

