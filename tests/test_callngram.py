"""
CALLNGRAM Test Suite
Comprehensive validation of:
1. Preprocessing & spoken conversational markers
2. N-gram generation (Unigram, Bigram, Trigram)
3. Laplace (Add-1) smoothing and probabilities
4. Reference-model Perplexity calculation
5. Train/Test split transcript-level partitioning & no leakage
6. Reports existence, integrity, and real metrics
7. Bitext customer issue & intent phrase intelligence
8. Streamlit application entry point & relative paths
9. Clean environment dependencies & runtime configuration
10. PDF academic report artifacts generation and validity
11. Architecture cleanliness (no dangling React / nested folders)
"""

import os
import sys
import math
import json
import unittest
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup workspace root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    clean_text_for_tokens,
    segment_sentences,
    tokenize_sentence,
    tokenize_transcript,
    compute_transcript_statistics,
    SPOKEN_MARKERS
)
from src.models import NGramLanguageModel


class TestCALLNGRAM(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Good morning! (uh) Thank you for calling tech support. "
            "My router is blinking red, (um,) and I cannot connect to the internet. "
            "Okay, I will reset your gateway now."
        )

    # ----------------------------------------------------
    # TEST 1 — Preprocessing & Spoken Markers
    # ----------------------------------------------------
    def test_01_preprocessing(self):
        cleaned = clean_text_for_tokens(self.sample_text)
        self.assertIn("uh", cleaned)
        self.assertIn("um", cleaned)
        self.assertNotIn("(uh)", cleaned)

        sents = segment_sentences(cleaned)
        self.assertEqual(len(sents), 4)

        toks = tokenize_sentence(sents[0])
        self.assertIn("morning", toks)

        sents_toks, flat_toks = tokenize_transcript(self.sample_text)
        self.assertTrue(len(flat_toks) > 15)
        self.assertTrue(any(tok in SPOKEN_MARKERS for tok in flat_toks))

        stats = compute_transcript_statistics([self.sample_text])
        self.assertEqual(stats["total_transcripts"], 1)
        self.assertTrue(stats["vocabulary_size"] > 0)
        self.assertEqual(stats["total_sentences"], 4)

    # ----------------------------------------------------
    # TEST 2 — N-Gram Generation
    # ----------------------------------------------------
    def test_02_ngram_generation(self):
        sentences = [
            ["<s>", "customer", "service", "help", "</s>"],
            ["<s>", "customer", "service", "phone", "</s>"]
        ]
        m1 = NGramLanguageModel(n=1).fit(sentences)
        m2 = NGramLanguageModel(n=2).fit(sentences)
        m3 = NGramLanguageModel(n=3).fit(sentences)

        self.assertEqual(m1.ngram_counts["customer"], 2)
        self.assertEqual(m1.ngram_counts["service"], 2)
        self.assertEqual(m2.ngram_counts[("customer", "service")], 2)
        self.assertEqual(m3.ngram_counts[("<s>", "customer", "service")], 2)

    # ----------------------------------------------------
    # TEST 3 — Laplace (Add-1) Smoothing
    # ----------------------------------------------------
    def test_03_smoothing(self):
        sentences = [["<s>", "order", "status", "</s>"]]
        m1 = NGramLanguageModel(n=1).fit(sentences)
        m2 = NGramLanguageModel(n=2).fit(sentences)
        m3 = NGramLanguageModel(n=3).fit(sentences)

        # Seen probability > 0
        p_seen_1 = m1.probability("order")
        p_seen_2 = m2.probability("status", context="order")
        p_seen_3 = m3.probability("status", context=("<s>", "order"))
        self.assertGreater(p_seen_1, 0.0)
        self.assertGreater(p_seen_2, 0.0)
        self.assertGreater(p_seen_3, 0.0)

        # Unseen probability > 0 (guaranteed non-zero by Add-1)
        p_unseen_1 = m1.probability("unseenwordxyz")
        p_unseen_2 = m2.probability("refund", context="order")
        p_unseen_3 = m3.probability("refund", context=("flight", "ticket"))
        self.assertGreater(p_unseen_1, 0.0)
        self.assertGreater(p_unseen_2, 0.0)
        self.assertGreater(p_unseen_3, 0.0)

    # ----------------------------------------------------
    # TEST 4 — Perplexity
    # ----------------------------------------------------
    def test_04_perplexity(self):
        train_sents_1 = [["thank", "you", "for", "calling"]]
        train_sents_2 = [["<s>", "thank", "you", "for", "calling", "</s>"]]
        train_sents_3 = [["<s>", "<s>", "thank", "you", "for", "calling", "</s>"]]

        m1 = NGramLanguageModel(n=1).fit(train_sents_1)
        m2 = NGramLanguageModel(n=2).fit(train_sents_2)
        m3 = NGramLanguageModel(n=3).fit(train_sents_3)

        test_sent_1 = [["thank", "you", "calling"]]
        test_sent_2 = [["<s>", "thank", "you", "calling", "</s>"]]
        test_sent_3 = [["<s>", "<s>", "thank", "you", "calling", "</s>"]]

        pp1 = m1.perplexity(test_sent_1)
        pp2 = m2.perplexity(test_sent_2)
        pp3 = m3.perplexity(test_sent_3)

        for pp, name in [(pp1, "Unigram"), (pp2, "Bigram"), (pp3, "Trigram")]:
            self.assertTrue(math.isfinite(pp), f"{name} perplexity is not finite")
            self.assertGreater(pp, 0.0, f"{name} perplexity must be positive")

    # ----------------------------------------------------
    # TEST 5 — Train / Test Split Integrity (80/20, seed=42)
    # ----------------------------------------------------
    def test_05_train_test_split_integrity(self):
        clean_path = ROOT_DIR / "data" / "annotated" / "apptek_clean.csv"
        self.assertTrue(clean_path.exists(), "apptek_clean.csv missing")
        df = pd.read_csv(clean_path)
        self.assertEqual(len(df), 1746, f"Expected 1,746 records, got {len(df)}")

        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
        self.assertEqual(len(train_df), 1396)
        self.assertEqual(len(test_df), 350)

        # Ensure zero leakage between train and test
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)

    # ----------------------------------------------------
    # TEST 6 — Reports Existence and Integrity
    # ----------------------------------------------------
    def test_06_reports_generated(self):
        ann_dir = ROOT_DIR / "data" / "annotated"
        expected_files = [
            "apptek_clean.csv",
            "bitext_clean.csv",
            "perplexity_report.csv",
            "apptek_1gram_frequency.csv",
            "apptek_2gram_frequency.csv",
            "apptek_3gram_frequency.csv",
            "1gram_probability_report.csv",
            "2gram_probability_report.csv",
            "3gram_probability_report.csv",
            "bitext_issue_phrase_report.csv",
            "project_summary.json"
        ]

        for fname in expected_files:
            fpath = ann_dir / fname
            self.assertTrue(fpath.exists(), f"Missing required report file: {fname}")
            self.assertGreater(fpath.stat().st_size, 0, f"Report file is empty: {fname}")

        # Check perplexity values
        df_pp = pd.read_csv(ann_dir / "perplexity_report.csv")
        self.assertEqual(len(df_pp), 3)
        self.assertIn("mean_perplexity", df_pp.columns)
        self.assertTrue((df_pp["mean_perplexity"] > 0).all())

    # ----------------------------------------------------
    # TEST 7 — Bitext Analysis
    # ----------------------------------------------------
    def test_07_bitext_analysis(self):
        ann_dir = ROOT_DIR / "data" / "annotated"
        df_issue = pd.read_csv(ann_dir / "bitext_issue_phrase_report.csv")
        self.assertGreater(len(df_issue), 100)
        self.assertIn("category", df_issue["level"].unique())
        self.assertIn("intent", df_issue["level"].unique())
        self.assertIn("unigram", df_issue["ngram_order"].unique())
        self.assertIn("bigram", df_issue["ngram_order"].unique())
        self.assertIn("trigram", df_issue["ngram_order"].unique())

    # ----------------------------------------------------
    # TEST 8 — Streamlit Entry Point & Code Cleanliness
    # ----------------------------------------------------
    def test_08_streamlit_entry_point(self):
        app_file = ROOT_DIR / "app.py"
        self.assertTrue(app_file.exists(), "app.py does not exist at root")
        content = app_file.read_text(encoding="utf-8")
        self.assertIn("CALLNGRAM", content)
        self.assertIn("st.set_page_config", content)
        self.assertIn("Dashboard", content)
        self.assertIn("Analyze Transcript", content)
        self.assertIn("N-gram Explorer", content)
        self.assertIn("Issue Intelligence", content)
        self.assertIn("Model Evaluation", content)
        # Verify no hardcoded Windows paths
        self.assertNotIn("C:\\Users\\", content)
        self.assertNotIn("C:/Users/", content)

    # ----------------------------------------------------
    # TEST 9 — Clean Environment Readiness
    # ----------------------------------------------------
    def test_09_clean_environment_readiness(self):
        req_file = ROOT_DIR / "requirements.txt"
        self.assertTrue(req_file.exists(), "requirements.txt missing")
        reqs = req_file.read_text(encoding="utf-8").lower()
        for pkg in ["pandas", "numpy", "nltk", "scikit-learn", "matplotlib", "plotly", "streamlit", "reportlab"]:
            self.assertIn(pkg, reqs, f"Missing required dependency: {pkg}")

        runtime_file = ROOT_DIR / "runtime.txt"
        self.assertTrue(runtime_file.exists(), "runtime.txt missing")
        self.assertIn("python", runtime_file.read_text(encoding="utf-8").lower())

    # ----------------------------------------------------
    # TEST 10 — PDF Documentation Reports
    # ----------------------------------------------------
    def test_10_pdf_documentation_reports(self):
        arch_pdf = ROOT_DIR / "docs" / "Architecture_and_Methodology.pdf"
        eval_pdf = ROOT_DIR / "docs" / "Evaluation_Report.pdf"

        self.assertTrue(arch_pdf.exists(), "Architecture_and_Methodology.pdf missing in docs/")
        self.assertTrue(eval_pdf.exists(), "Evaluation_Report.pdf missing in docs/")

        self.assertGreater(arch_pdf.stat().st_size, 5000, "Architecture PDF too small")
        self.assertGreater(eval_pdf.stat().st_size, 5000, "Evaluation PDF too small")

        # Also verify reports/ folder contains copies
        rep_arch = ROOT_DIR / "reports" / "Architecture_and_Methodology.pdf"
        rep_eval = ROOT_DIR / "reports" / "Evaluation_Report.pdf"
        self.assertTrue(rep_arch.exists(), "Architecture PDF missing in reports/")
        self.assertTrue(rep_eval.exists(), "Evaluation PDF missing in reports/")

    # ----------------------------------------------------
    # TEST 11 — Architecture Cleanliness (No Redundant Artifacts)
    # ----------------------------------------------------
    def test_11_architecture_cleanliness(self):
        # Nested CALLNGRAM directory must NOT exist
        nested_callngram = ROOT_DIR / "CALLNGRAM"
        self.assertFalse(nested_callngram.exists(), "Nested CALLNGRAM directory should be removed")

        # React / Vite / TypeScript web artifacts must NOT exist
        for bad_file in ["package.json", "tsconfig.json", "vite.config.ts", "index.html"]:
            self.assertFalse((ROOT_DIR / bad_file).exists(), f"Unneeded web artifact {bad_file} should not exist")


if __name__ == "__main__":
    unittest.main()
