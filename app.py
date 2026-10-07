"""
CALLNGRAM — Customer Call Language Analytics via N-gram Language Modeling
B.Tech AIML Mini-Project Application

Features:
- Primary Corpus (AppTek): 1,746 transcripts (80/20 train/test split, seed 42)
- Secondary Corpus (Bitext): 26,872 customer support queries
- Unigram, Bigram, Trigram models with Laplace (Add-1) smoothing
- Real-time transcript perplexity evaluation against trained reference models
- N-gram frequency and probability exploration
- Context Probability Lookup with Bigram & Trigram next-word prediction
- Customer issue & intent phrase discovery (Category -> Intent -> Top phrases)
- Academic documentation artifacts (PDF)
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

# Setup base directory and import path
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "src").exists():
    sys.path.insert(0, str(BASE_DIR))
elif (BASE_DIR.parent / "src").exists():
    sys.path.insert(0, str(BASE_DIR.parent))

from src.preprocessing import (
    clean_text_for_tokens,
    segment_sentences,
    tokenize_sentence,
    tokenize_transcript,
    SPOKEN_MARKERS
)
from src.models import NGramLanguageModel

# Streamlit Page Configuration
st.set_page_config(
    page_title="CALLNGRAM — Customer Call Language Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme & Professional Styling
st.markdown("""
<style>
    /* Main Layout & Base Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Academic Header Styles */
    .project-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: #FFFFFF;
        padding: 24px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .project-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        color: #FFFFFF;
    }
    .project-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .project-badge-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .project-badge {
        background: rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 20px;
        margin-bottom: 4px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
    }
    .section-caption {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 16px;
    }

    /* Metric Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .stat-detail {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Academic Callout Boxes */
    .academic-box {
        background-color: #FFFFFF;
        border-left: 4px solid #1E3A8A;
        border-right: 1px solid #CBD5E1;
        border-top: 1px solid #CBD5E1;
        border-bottom: 1px solid #CBD5E1;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #1E293B;
    }

    /* Methodology Notes */
    .viva-note {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        color: #78350F;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        margin: 12px 0;
    }

    /* Pipeline Diagram Styling */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        margin: 18px 0 24px 0;
    }
    .pipeline-step {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #1E3A8A;
        border-radius: 8px;
        padding: 12px 18px;
        width: 100%;
        max-width: 840px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .pipeline-step-num {
        background: #1E3A8A;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .pipeline-step-body {
        flex-grow: 1;
    }
    .pipeline-step-title {
        color: #0F172A;
        font-weight: 700;
        font-size: 1.0rem;
    }
    .pipeline-step-desc {
        color: #475569;
        font-size: 0.86rem;
        margin-top: 2px;
    }
    .pipeline-arrow {
        font-size: 1.3rem;
        color: #2563EB;
        font-weight: 900;
        line-height: 1;
        user-select: none;
    }

    /* High-contrast Radio and Selectbox Labels */
    div[data-testid="stRadio"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_project_artifacts():
    """Loads all pipeline artifacts and summary metadata."""
    ann_dir = BASE_DIR / "data" / "annotated"
    if not ann_dir.exists() or not (ann_dir / "project_summary.json").exists():
        ann_dir = BASE_DIR.parent / "data" / "annotated"
        if not ann_dir.exists() or not (ann_dir / "project_summary.json").exists():
            return None

    artifacts = {}
    try:
        with open(ann_dir / "project_summary.json", "r", encoding="utf-8") as f:
            artifacts["summary"] = json.load(f)

        artifacts["perplexity"] = pd.read_csv(ann_dir / "perplexity_report.csv")
        artifacts["freq_1g"] = pd.read_csv(ann_dir / "apptek_1gram_frequency.csv")
        artifacts["freq_2g"] = pd.read_csv(ann_dir / "apptek_2gram_frequency.csv")
        artifacts["freq_3g"] = pd.read_csv(ann_dir / "apptek_3gram_frequency.csv")

        artifacts["prob_1g"] = pd.read_csv(ann_dir / "1gram_probability_report.csv")
        artifacts["prob_2g"] = pd.read_csv(ann_dir / "2gram_probability_report.csv")
        artifacts["prob_3g"] = pd.read_csv(ann_dir / "3gram_probability_report.csv")

        artifacts["issue_report"] = pd.read_csv(ann_dir / "bitext_issue_phrase_report.csv")
    except Exception as e:
        st.error(f"Error loading pipeline artifacts: {e}")
        return None

    return artifacts


@st.cache_resource
def load_trained_models():
    """
    Trains in-memory reference models on the exact 80% training partition (N=1,396, seed=42)
    of the AppTek corpus. Guarantees 100% mathematical consistency with offline evaluation.
    """
    clean_path = BASE_DIR / "data" / "annotated" / "apptek_clean.csv"
    if not clean_path.exists():
        clean_path = BASE_DIR.parent / "data" / "annotated" / "apptek_clean.csv"
        if not clean_path.exists():
            return None, None, None, None, None

    try:
        df = pd.read_csv(clean_path)
        # Exact reproducible 80/20 train/test split matching the pipeline
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
        train_texts = train_df["clean_text"].dropna().tolist()

        sents_1 = []
        sents_2 = []
        sents_3 = []
        vocab = set()

        for t in train_texts:
            sents = segment_sentences(str(t))
            for s in sents:
                toks = tokenize_sentence(s, lowercase=True)
                if not toks:
                    continue
                vocab.update(toks)
                sents_1.append(toks)
                sents_2.append(["<s>"] + toks + ["</s>"])
                sents_3.append(["<s>", "<s>"] + toks + ["</s>"])

        m1 = NGramLanguageModel(n=1).fit(sents_1, vocab_override=vocab)
        m2 = NGramLanguageModel(n=2).fit(sents_2, vocab_override=vocab)
        m3 = NGramLanguageModel(n=3).fit(sents_3, vocab_override=vocab)
        return m1, m2, m3, train_df, test_df
    except Exception as e:
        st.error(f"Error fitting reference models: {e}")
        return None, None, None, None, None


artifacts = load_project_artifacts()
m1_ref, m2_ref, m3_ref, train_df, test_df = load_trained_models()

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <h2 style="color: #FFFFFF; margin: 0; font-size: 1.5rem; font-weight: 800;">📞 CALLNGRAM</h2>
    <p style="color: #94A3B8; font-size: 0.85rem; margin: 2px 0 0 0;">Customer Call Language Analytics</p>
    <p style="color: #64748B; font-size: 0.75rem; margin: 2px 0 0 0;">B.Tech AIML Mini-Project</p>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Modules",
    [
        "Dashboard",
        "Analyze Transcript",
        "N-gram Explorer",
        "Issue Intelligence",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Technical Setup**")
st.sidebar.markdown("• **Primary Corpus**: AppTek Dialogues\n• **Secondary Corpus**: Bitext 27K Inquiries\n• **Smoothing**: Laplace (Add-1)\n• **Split**: 80% Train / 20% Held-Out")
st.sidebar.caption("Evaluation Metric: Cross-Entropy Perplexity")

# ----------------------------------------------------
# 1. DASHBOARD
# ----------------------------------------------------
if nav_choice == "Dashboard":
    st.markdown("""
    <div class="project-header">
        <div class="project-title">CALLNGRAM</div>
        <div class="project-subtitle">Customer Call Language Analytics via N-gram Language Modeling</div>
        <div class="project-badge-row">
            <span class="project-badge">Unigram (N=1)</span>
            <span class="project-badge">Bigram (N=2)</span>
            <span class="project-badge">Trigram (N=3)</span>
            <span class="project-badge">Laplace (Add-1) Smoothing</span>
            <span class="project-badge">Perplexity Evaluation</span>
            <span class="project-badge">Customer Issue Mining</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if artifacts and "summary" in artifacts:
        s = artifacts["summary"]
        app = s.get("apptek", {})
        bit = s.get("bitext", {})
        models_eval = s.get("models_evaluated", [])

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{app.get('total_transcripts', 1746):,}</div>
                <div class="stat-label">Total Transcripts</div>
                <div class="stat-detail">Primary AppTek Corpus</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{app.get('train_transcripts', 1396)} / {app.get('test_transcripts', 350)}</div>
                <div class="stat-label">Train / Test Split</div>
                <div class="stat-detail">80% / 20% Held-Out (Seed 42)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{app.get('vocabulary_size', 17868):,}</div>
                <div class="stat-label">Corpus Vocabulary</div>
                <div class="stat-detail">16,186 Training Lexicon (|V|)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{bit.get('total_examples', 26872):,}</div>
                <div class="stat-label">Support Inquiries</div>
                <div class="stat-detail">{bit.get('categories_count', 11)} Categories • {bit.get('intents_count', 27)} Intents</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Corpus Architecture (Separately Maintained)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Strict separation ensures continuous dialogue modeling without synthetic distribution distortion.</div>', unsafe_allow_html=True)

        c_app, c_bit = st.columns(2)
        with c_app:
            st.markdown("""
            <div class="academic-box">
                <b style="color: #1E3A8A; font-size: 1rem;">Primary Corpus: AppTek Call-Center Dialogues</b><br/>
                <b>Nature:</b> Conversational call-center-style dialogue transcripts featuring agent and customer roles across multi-turn telephone interactions.<br/>
                <b>Volume:</b> 1,746 transcripts (873 customer, 873 agent dialogues).<br/>
                <b>Diversity:</b> 16 business domains (Banking, Tech, Delivery, Telecom, Travel, Health, etc.) across 14 global English accents.<br/>
                <b>Total Tokens:</b> ~1.27 Million words (average 730.1 words per call).<br/>
                <b>Role in Project:</b> Used for text preprocessing, training Unigram/Bigram/Trigram models, Add-1 smoothing, and held-out test perplexity evaluation.
            </div>
            """, unsafe_allow_html=True)
        with c_bit:
            st.markdown("""
            <div class="academic-box">
                <b style="color: #1E3A8A; font-size: 1rem;">Secondary Corpus: Bitext Customer Support Dataset</b><br/>
                <b>Nature:</b> Curated customer service inquiry queries labeled with specific intent and issue category taxonomies.<br/>
                <b>Volume:</b> 26,872 customer utterances.<br/>
                <b>Taxonomy:</b> 11 commercial issue categories and 27 granular customer intents.<br/>
                <b>Average Length:</b> ~12 words per utterance.<br/>
                <b>Role in Project:</b> Used exclusively for customer issue phrase discovery, recurring domain terminology extraction, and intent phrase profiling.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Language Modeling Pipeline Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">End-to-end mathematical workflow from conversational speech transcripts to perplexity and issue discovery.</div>', unsafe_allow_html=True)

        # Clear, responsive, and robust HTML/CSS pipeline visualization
        st.markdown("""
        <div class="pipeline-container">
            <div class="pipeline-step">
                <div class="pipeline-step-num">1</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Raw Call Transcript</div>
                    <div class="pipeline-step-desc">1,746 AppTek call-center-style dialogue transcripts (873 customer, 873 agent dialogues) across 16 service domains.</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">2</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Preprocessing</div>
                    <div class="pipeline-step-desc">Case lowercasing, bracket normalization, whitespace collapse, and explicit preservation of spoken hesitation markers (<i>uh, um, okay, yeah, mhm</i>).</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">3</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Tokenization</div>
                    <div class="pipeline-step-desc">Sentence segmentation with offline regex fallback, word tokenization, and sentence boundary token injection (<code>&lt;s&gt;</code> and <code>&lt;/s&gt;</code>).</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">4</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">N-gram Generation</div>
                    <div class="pipeline-step-desc">Extraction of Unigrams (single words), Bigrams (two-word collocations), and Trigrams (three-word collocations) via sliding windows.</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">5</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Laplace Smoothing</div>
                    <div class="pipeline-step-desc">Application of Add-1 smoothing with fixed training vocabulary |V| = 16,186 (effective |V| = 16,187) to prevent zero-probability crashes on unseen transitions.</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">6</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Probability Calculation</div>
                    <div class="pipeline-step-desc">Estimation of transition probabilities: P(w|context) = (count(context, w) + 1) / (count(context) + |V|) alongside unsmoothed MLE baseline.</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">7</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Perplexity</div>
                    <div class="pipeline-step-desc">Evaluation of intrinsic predictive uncertainty PP(W) = exp(-1/M &Sigma; ln P) across 350 held-out test transcripts against reference models.</div>
                </div>
            </div>
            <div class="pipeline-arrow">&#x2193;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-num">8</div>
                <div class="pipeline-step-body">
                    <div class="pipeline-step-title">Issue Phrase Analysis</div>
                    <div class="pipeline-step-desc">Extraction of recurring multi-word collocations across 11 customer support categories and 27 intents in the secondary Bitext corpus.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model Benchmark Snapshot
        st.markdown('<div class="section-title">Reference Language Model Benchmarks</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Evaluated across all 350 held-out AppTek transcripts under Laplace (Add-1) smoothing.</div>', unsafe_allow_html=True)

        df_bench = pd.DataFrame(models_eval)
        if not df_bench.empty:
            b_col1, b_col2 = st.columns([3, 2])
            with b_col1:
                fig_bench = px.bar(
                    df_bench,
                    x="model",
                    y="mean_perplexity",
                    text="mean_perplexity",
                    color="model",
                    color_discrete_sequence=["#1E3A8A", "#2563EB", "#D97706"],
                    labels={"model": "Model", "mean_perplexity": "Mean Perplexity"}
                )
                fig_bench.update_traces(textposition='outside')
                fig_bench.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_bench, use_container_width=True)
            with b_col2:
                st.dataframe(df_bench, use_container_width=True, height=280)

# ----------------------------------------------------
# 2. ANALYZE TRANSCRIPT
# ----------------------------------------------------
elif nav_choice == "Analyze Transcript":
    st.markdown('<div class="section-title">Transcript Analysis & Perplexity Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Evaluate text properties, extract multi-word collocations, and calculate sequence perplexity against trained AppTek reference models.</div>', unsafe_allow_html=True)

    # Preset Sample Selector
    sample_choice = st.selectbox(
        "Select Transcript Input Source",
        [
            "Sample Transcript 1: Order Delay & Refund Inquiry (Illustrative)",
            "Sample Transcript 2: Technical Troubleshooting & Router Reset (Illustrative)",
            "Genuine Held-Out Test Transcript from AppTek Corpus (Retail Customer)",
            "Custom User Input / File Upload"
        ]
    )

    sample_1_text = (
        "Good morning, thank you for calling customer service. This is Sarah speaking, how can I help you today? "
        "Hi Sarah, uh I am calling because I have not received my delivery yet and the tracking order page says "
        "it was delayed. Can you help me check the status or get a refund please? "
        "Of course, um let me look up your order number right away. Okay, thank you so much."
    )
    sample_2_text = (
        "Hello tech support, thank you for taking my call. My home internet gateway has been blinking red all morning. "
        "I tried restarting the router (uh) but the light remains red and I cannot connect to my work laptop. "
        "Okay, um let me perform a remote diagnostic test on your broadband line right now."
    )

    chosen_default = ""
    source_label = "Custom Input"
    if "Sample Transcript 1" in sample_choice:
        chosen_default = sample_1_text
        source_label = "Sample Transcript (Illustrative Customer Service Inquiry)"
    elif "Sample Transcript 2" in sample_choice:
        chosen_default = sample_2_text
        source_label = "Sample Transcript (Illustrative Technical Support Call)"
    elif "Genuine Held-Out" in sample_choice and test_df is not None and not test_df.empty:
        real_rec = test_df.iloc[0]
        chosen_default = str(real_rec["clean_text"])
        source_label = f"Held-Out AppTek Test Dialogue (Domain: {real_rec.get('domain', 'retail')}, Role: {real_rec.get('role', 'customer')})"

    st.info(f"**Current Input Source:** `{source_label}`")

    # File uploader
    upload_file = st.file_uploader("Or Upload Custom Transcript (.txt)", type=["txt"])
    if upload_file is not None:
        try:
            chosen_default = upload_file.read().decode("utf-8")
            source_label = f"Uploaded File: {upload_file.name}"
        except Exception as e:
            st.error(f"Error reading file: {e}")

    user_text = st.text_area(
        "Transcript Content",
        value=chosen_default,
        height=180,
        placeholder="Please enter or upload a transcript."
    )

    if st.button("🚀 Analyze Transcript & Compute Perplexity", type="primary"):
        if not user_text.strip():
            st.warning("Please enter or upload a transcript.")
        else:
            with st.spinner("Tokenizing, scoring sequence, and extracting N-grams..."):
                cleaned = clean_text_for_tokens(user_text)
                sentences = segment_sentences(cleaned)
                all_tokens_nested = [tokenize_sentence(s, lowercase=True) for s in sentences]
                all_words = [tok for s in all_tokens_nested for tok in s]
                total_word_count = len(all_words)

                # Count spoken markers
                marker_counts = {m: all_words.count(m) for m in SPOKEN_MARKERS if m in all_words}
                total_markers = sum(marker_counts.values())

                st.markdown('<div class="section-title">1. Transcript Characteristics</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Sentence Count", len(sentences))
                with m2:
                    st.metric("Total Tokens", total_word_count)
                with m3:
                    st.metric("Unique Words (Vocab)", len(set(all_words)))
                with m4:
                    st.metric("Spoken Markers", f"{total_markers} ({total_markers/max(total_word_count,1)*100:.1f}%)")

                if marker_counts:
                    st.caption(f"**Detected Conversational Hesitation Tokens:** {', '.join([f'{k} ({v})' for k, v in marker_counts.items()])}")

                # Perplexity against reference models
                st.markdown('<div class="section-title">2. Sequence Predictability & Perplexity Scoring</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-caption">Evaluated strictly against the reference models trained on 80% AppTek training data.</div>', unsafe_allow_html=True)

                if m1_ref is not None and m2_ref is not None and m3_ref is not None:
                    # Score against reference models
                    pp_u = m1_ref.perplexity(all_tokens_nested)
                    pp_b = m2_ref.perplexity(all_tokens_nested)
                    pp_t = m3_ref.perplexity(all_tokens_nested)

                    p_col1, p_col2, p_col3 = st.columns(3)
                    with p_col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp_u:,.2f}</div>
                            <div class="stat-label">Unigram Perplexity</div>
                            <div class="stat-detail">Reference Corpus Baseline: 551.93</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with p_col2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp_b:,.2f}</div>
                            <div class="stat-label">Bigram Perplexity</div>
                            <div class="stat-detail">Reference Corpus Baseline: 586.42</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with p_col3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp_t:,.2f}</div>
                            <div class="stat-label">Trigram Perplexity</div>
                            <div class="stat-detail">Reference Corpus Baseline: 2,845.61</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="academic-box">
                        <b>Mathematical Interpretation:</b> Perplexity reflects the effective branching factor under the model.
                        A lower score indicates that the sequence aligns closely with learned conversational patterns.
                        Trigram perplexity is higher under Add-1 smoothing because multi-word parameter space sparsity (|V|² ≈ 2.62 × 10⁸)
                        subjects unseen test transitions to severe uniform probability discounts.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Reference models could not be loaded. Please ensure dataset files exist.")

                # Extract local n-grams
                def extract_local_ngrams(tokens: List[str], n: int):
                    if len(tokens) < n:
                        return []
                    counts = {}
                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i:i+n])
                        counts[ngram] = counts.get(ngram, 0) + 1
                    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    return [(" ".join(k) if n > 1 else k[0], v) for k, v in sorted_items]

                u_local = extract_local_ngrams(all_words, 1)
                b_local = extract_local_ngrams(all_words, 2)
                t_local = extract_local_ngrams(all_words, 3)

                st.markdown('<div class="section-title">3. Extracted Transcript Collocations</div>', unsafe_allow_html=True)
                tab_u, tab_b, tab_t, tab_toks = st.tabs(["Top Unigrams", "Top Bigrams", "Top Trigrams", "Sentence Boundary View"])
                with tab_u:
                    df_u = pd.DataFrame(u_local[:20], columns=["Unigram", "Count"])
                    st.dataframe(df_u, use_container_width=True)
                with tab_b:
                    df_b = pd.DataFrame(b_local[:20], columns=["Bigram Phrase", "Count"])
                    st.dataframe(df_b, use_container_width=True)
                with tab_t:
                    df_t = pd.DataFrame(t_local[:20], columns=["Trigram Phrase", "Count"])
                    st.dataframe(df_t, use_container_width=True)
                with tab_toks:
                    st.caption("Displays segmented sentences with injected boundary tags for conditional modeling:")
                    for idx, s in enumerate(sentences[:6]):
                        toks_w_bound = ["<s>"] + tokenize_sentence(s) + ["</s>"]
                        st.code(f"Sentence {idx + 1}: {' '.join(toks_w_bound)}")

                # Export CSV
                all_export = []
                for p, c in u_local: all_export.append({"Order": "Unigram", "Phrase": p, "Count": c})
                for p, c in b_local: all_export.append({"Order": "Bigram", "Phrase": p, "Count": c})
                for p, c in t_local: all_export.append({"Order": "Trigram", "Phrase": p, "Count": c})
                csv_bytes = pd.DataFrame(all_export).to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Transcript N-grams (CSV)", csv_bytes, "transcript_ngrams.csv", "text/csv")

# ----------------------------------------------------
# 3. N-GRAM EXPLORER
# ----------------------------------------------------
elif nav_choice == "N-gram Explorer":
    st.markdown('<div class="section-title">N-gram Frequency & Probability Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Inspect vocabulary distributions, multi-word collocations, and Laplace smoothed transition probabilities from the primary training corpus.</div>', unsafe_allow_html=True)

    if not artifacts:
        st.error("Annotated reports not found. Run `python src/pipeline.py` first.")
    else:
        # Explicit, high-contrast radio selector
        order = st.radio(
            "Select N-gram Order",
            ["Unigram (Single Words)", "Bigram (Two-Word Collocations)", "Trigram (Three-Word Sequences)"],
            horizontal=True
        )

        if "Unigram" in order:
            df_curr = artifacts.get("prob_1g", artifacts.get("freq_1g"))
            order_name = "Unigram"
            title = "Unigram Distribution (Single Words)"
        elif "Bigram" in order:
            df_curr = artifacts.get("prob_2g", artifacts.get("freq_2g"))
            order_name = "Bigram"
            title = "Bigram Distribution (Two-Word Collocations)"
        else:
            df_curr = artifacts.get("prob_3g", artifacts.get("freq_3g"))
            order_name = "Trigram"
            title = "Trigram Distribution (Three-Word Sequences)"

        if df_curr is not None and not df_curr.empty:
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                search_query = st.text_input("Filter phrases by keyword (e.g., 'thank', 'service', 'account', 'order')", "")
            with f_col2:
                top_limit = st.selectbox("Number of results to display", [10, 25, 50, 100], index=1)

            if search_query:
                filtered_df = df_curr[df_curr["ngram"].astype(str).str.contains(search_query.lower(), na=False)].copy()
            else:
                filtered_df = df_curr.copy()

            st.write(f"Displaying **{min(len(filtered_df), top_limit)}** of **{len(filtered_df):,}** phrases:")

            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                chart_df = filtered_df.head(top_limit).copy()
                fig = px.bar(
                    chart_df,
                    x="count",
                    y="ngram",
                    orientation="h",
                    title=f"Top {min(len(filtered_df), top_limit)} Phrases — {title}",
                    color="count",
                    color_continuous_scale="Blues",
                    labels={"ngram": "Phrase", "count": "Corpus Frequency"}
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=520,
                    margin=dict(l=10, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                # Table with exact requested columns: N-gram | Count | Probability
                table_display = pd.DataFrame()
                table_display["N-gram"] = filtered_df["ngram"]
                table_display["Count"] = filtered_df["count"].map("{:,}".format)
                if "probability" in filtered_df.columns:
                    table_display["Probability"] = filtered_df["probability"].map("{:.6f}".format)
                else:
                    table_display["Probability"] = "N/A"

                st.dataframe(table_display.head(top_limit), use_container_width=True, height=520)

            # CSV download
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download {order_name} Frequency Report (CSV)",
                csv_data,
                f"apptek_{order_name.lower()}_report.csv",
                "text/csv"
            )

        # ----------------------------------------------------
        # Context Probability Lookup
        # ----------------------------------------------------
        st.markdown('<div class="section-title">Context Probability Lookup</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Calculate Laplace-smoothed transition probabilities for a word given its preceding context.</div>', unsafe_allow_html=True)

        lookup_model_type = st.radio(
            "Model Order for Context Lookup",
            ["Bigram Model: P(w_i | w_{i-1})", "Trigram Model: P(w_i | w_{i-2}, w_{i-1})"],
            horizontal=True
        )

        if "Bigram" in lookup_model_type:
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                ctx_word = st.text_input("Context Word (w_{i-1})", "thank")
            with c_in2:
                target_word = st.text_input("Possible Next Word (w_i)", "you")

            w1 = ctx_word.strip().lower()
            w2 = target_word.strip().lower()

            if not w1:
                st.warning("Please enter a preceding context word (e.g., 'thank', 'credit', 'customer').")
            elif not w2:
                st.warning("Please enter a candidate next word (e.g., 'you', 'card', 'service').")
            elif m2_ref is not None:
                # Real calculated values from reference model
                c_ctx = m2_ref.context_counts.get(w1, 0)
                c_ngram = m2_ref.ngram_counts.get((w1, w2), 0)
                v = m2_ref.vocab_size
                p_smooth = m2_ref.probability(w2, context=w1, smoothed=True)
                p_mle = m2_ref.mle_probability(w2, context=w1)

                st.markdown(f"### Result: `P({w2} | {w1})`")

                # Formula box with values substituted
                st.markdown(f"""
                <div class="academic-box">
                    <b>Laplace Smoothing Formula:</b><br/>
                    $$P({w2} \\mid {w1}) = \\frac{{C({w1}, {w2}) + 1}}{{C({w1}) + |V|}} = \\frac{{{c_ngram} + 1}}{{{c_ctx} + {v}}} = {p_smooth:.6f}$$
                    <br/>
                    <b>Unsmoothed Maximum Likelihood Estimate (MLE):</b><br/>
                    $$P_{{MLE}}({w2} \\mid {w1}) = \\frac{{C({w1}, {w2})}}{{C({w1})}} = {p_mle:.6f}$$
                </div>
                """, unsafe_allow_html=True)

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Context Count C(w_{i-1})", f"{c_ctx:,}")
                with k2:
                    st.metric("Bigram Count C(w_{i-1}, w_i)", f"{c_ngram:,}")
                with k3:
                    st.metric("Laplace Smoothed P", f"{p_smooth:.6f}")
                with k4:
                    st.metric("MLE Probability", f"{p_mle:.6f}")

                # Transitions list
                st.markdown(f"#### Top Observed Continuations for: `{w1}`")
                transitions = m2_ref.get_transitions(w1, top_k=10)
                if transitions:
                    t_df = pd.DataFrame(transitions)
                    t_df.columns = ["Next Word", "Count", "Laplace Probability", "MLE Probability"]
                    t_df["Laplace Probability"] = t_df["Laplace Probability"].map("{:.6f}".format)
                    t_df["MLE Probability"] = t_df["MLE Probability"].map("{:.6f}".format)
                    
                    t_col1, t_col2 = st.columns([3, 2])
                    with t_col1:
                        fig_trans = px.bar(
                            pd.DataFrame(transitions),
                            x="count",
                            y="next_word",
                            orientation="h",
                            title=f"Most Frequent Next Tokens after '{w1}'",
                            labels={"next_word": "Next Word", "count": "Co-occurrence Count"},
                            color="count",
                            color_continuous_scale="Blues"
                        )
                        fig_trans.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(l=10, r=20, t=30, b=20))
                        st.plotly_chart(fig_trans, use_container_width=True)
                    with t_col2:
                        st.dataframe(t_df, use_container_width=True, height=320)
                else:
                    st.info(f"Context `{w1}` was not observed in the training partition (C = 0). Under Add-1 smoothing, any candidate target token receives non-zero probability: P = 1 / (0 + {v}) = {1.0/v:.6f}.")

        else:
            # Trigram Model
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                ctx_phrase = st.text_input("Context History (Two Words: w_{i-2} w_{i-1})", "thank you")
            with c_in2:
                target_word = st.text_input("Possible Next Word (w_i)", "for")

            tokens = ctx_phrase.strip().lower().split()
            w3 = target_word.strip().lower()

            if len(tokens) < 2:
                st.warning("Please provide a 2-word context history for the Trigram model (e.g., 'thank you', 'customer service', 'credit card').")
            elif not w3:
                st.warning("Please enter a candidate next word (e.g., 'for', 'representative', 'number').")
            elif m3_ref is not None:
                ctx_tuple = (tokens[-2], tokens[-1])
                c_ctx = m3_ref.context_counts.get(ctx_tuple, 0)
                c_ngram = m3_ref.ngram_counts.get((ctx_tuple[0], ctx_tuple[1], w3), 0)
                v = m3_ref.vocab_size
                p_smooth = m3_ref.probability(w3, context=ctx_tuple, smoothed=True)
                p_mle = m3_ref.mle_probability(w3, context=ctx_tuple)

                st.markdown(f"### Result: `P({w3} | {ctx_tuple[0]}, {ctx_tuple[1]})`")

                # Formula box with values substituted
                st.markdown(f"""
                <div class="academic-box">
                    <b>Laplace Smoothing Formula:</b><br/>
                    $$P({w3} \\mid {ctx_tuple[0]}, {ctx_tuple[1]}) = \\frac{{C({ctx_tuple[0]}, {ctx_tuple[1]}, {w3}) + 1}}{{C({ctx_tuple[0]}, {ctx_tuple[1]}) + |V|}} = \\frac{{{c_ngram} + 1}}{{{c_ctx} + {v}}} = {p_smooth:.6f}$$
                    <br/>
                    <b>Unsmoothed Maximum Likelihood Estimate (MLE):</b><br/>
                    $$P_{{MLE}}({w3} \\mid {ctx_tuple[0]}, {ctx_tuple[1]}) = \\frac{{C({ctx_tuple[0]}, {ctx_tuple[1]}, {w3})}}{{C({ctx_tuple[0]}, {ctx_tuple[1]})}} = {p_mle:.6f}$$
                </div>
                """, unsafe_allow_html=True)

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Context Count C(w_{i-2}, w_{i-1})", f"{c_ctx:,}")
                with k2:
                    st.metric("Trigram Count C(ctx, w_i)", f"{c_ngram:,}")
                with k3:
                    st.metric("Laplace Smoothed P", f"{p_smooth:.6f}")
                with k4:
                    st.metric("MLE Probability", f"{p_mle:.6f}")

                # Transitions list
                st.markdown(f"#### Top Observed Continuations for: `{' '.join(ctx_tuple)}`")
                transitions = m3_ref.get_transitions(ctx_tuple, top_k=10)
                if transitions:
                    t_df = pd.DataFrame(transitions)
                    t_df.columns = ["Next Word", "Count", "Laplace Probability", "MLE Probability"]
                    t_df["Laplace Probability"] = t_df["Laplace Probability"].map("{:.6f}".format)
                    t_df["MLE Probability"] = t_df["MLE Probability"].map("{:.6f}".format)

                    t_col1, t_col2 = st.columns([3, 2])
                    with t_col1:
                        fig_trans = px.bar(
                            pd.DataFrame(transitions),
                            x="count",
                            y="next_word",
                            orientation="h",
                            title=f"Most Frequent Next Tokens after '{' '.join(ctx_tuple)}'",
                            labels={"next_word": "Next Word", "count": "Co-occurrence Count"},
                            color="count",
                            color_continuous_scale="Blues"
                        )
                        fig_trans.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(l=10, r=20, t=30, b=20))
                        st.plotly_chart(fig_trans, use_container_width=True)
                    with t_col2:
                        st.dataframe(t_df, use_container_width=True, height=320)
                else:
                    st.info(f"Context `{' '.join(ctx_tuple)}` was not observed in the training partition (C = 0). Under Add-1 smoothing, any candidate target token receives non-zero probability: P = 1 / (0 + {v}) = {1.0/v:.6f}.")

# ----------------------------------------------------
# 4. ISSUE INTELLIGENCE
# ----------------------------------------------------
elif nav_choice == "Issue Intelligence":
    st.markdown('<div class="section-title">Customer Issue Phrase Analysis</div>', unsafe_allow_html=True)

    # Concise methodology note
    st.markdown("""
    <div class="viva-note">
        Recurring N-grams are shown as language patterns associated with customer-support categories and intents.
        They do not constitute automatic intent classification by themselves.
    </div>
    """, unsafe_allow_html=True)

    if not artifacts or "issue_report" not in artifacts:
        st.error("Bitext issue phrase report not found. Run `python src/pipeline.py` first.")
    else:
        df_issue = artifacts["issue_report"]

        # Category -> Intent Mapping
        CATEGORY_INTENT_MAP = {
            'ORDER': ['cancel_order', 'change_order', 'place_order', 'track_order'],
            'REFUND': ['check_refund_policy', 'get_refund', 'track_refund'],
            'PAYMENT': ['check_payment_methods', 'payment_issue'],
            'ACCOUNT': ['create_account', 'delete_account', 'edit_account', 'recover_password', 'registration_problems', 'switch_account'],
            'CANCEL': ['check_cancellation_fee'],
            'DELIVERY': ['delivery_options', 'delivery_period'],
            'SHIPPING': ['change_shipping_address', 'set_up_shipping_address'],
            'INVOICE': ['check_invoice', 'get_invoice'],
            'CONTACT': ['contact_customer_service', 'contact_human_agent'],
            'FEEDBACK': ['complaint', 'review'],
            'SUBSCRIPTION': ['newsletter_subscription']
        }

        all_categories = sorted(list(CATEGORY_INTENT_MAP.keys()))
        
        sel_c1, sel_c2 = st.columns(2)
        with sel_c1:
            selected_cat = st.selectbox("1. Select Support Category", all_categories, index=0)

        with sel_c2:
            available_intents = ["All Category Phrases"] + CATEGORY_INTENT_MAP.get(selected_cat, [])
            selected_intent = st.selectbox("2. Select Intent within Category", available_intents, index=0)

        # Filter phrases based on selection
        if selected_intent == "All Category Phrases":
            df_sel = df_issue[(df_issue["level"] == "category") & (df_issue["target"] == selected_cat)].copy()
            view_label = f"Category: {selected_cat}"
        else:
            df_sel = df_issue[(df_issue["level"] == "intent") & (df_issue["target"] == selected_intent)].copy()
            view_label = f"Intent: {selected_intent} ({selected_cat})"

        st.markdown(f"### Recurring Phrases for: `{view_label}`")

        # Two columns for Bigrams & Trigrams
        col_b, col_t = st.columns(2)
        with col_b:
            st.markdown("**Top Bigram Issue Phrases**")
            b_df = df_sel[df_sel["ngram_order"] == "bigram"][["phrase", "count"]].reset_index(drop=True)
            b_df.columns = ["Bigram Phrase", "Count"]
            st.dataframe(b_df, use_container_width=True, height=280)

        with col_t:
            st.markdown("**Top Trigram Issue Phrases**")
            t_df = df_sel[df_sel["ngram_order"] == "trigram"][["phrase", "count"]].reset_index(drop=True)
            t_df.columns = ["Trigram Phrase", "Count"]
            st.dataframe(t_df, use_container_width=True, height=280)

        # Horizontal Bar Chart for readability
        chart_data = df_sel[df_sel["ngram_order"].isin(["bigram", "trigram"])].sort_values(by="count", ascending=True).tail(15)
        if not chart_data.empty:
            fig_issue = px.bar(
                chart_data,
                x="count",
                y="phrase",
                orientation="h",
                color="ngram_order",
                title=f"Top Collocations — {view_label}",
                color_discrete_sequence=["#2563EB", "#059669"],
                labels={"phrase": "Phrase", "count": "Corpus Frequency", "ngram_order": "Order"}
            )
            fig_issue.update_layout(height=400, margin=dict(l=10, r=20, t=30, b=20))
            st.plotly_chart(fig_issue, use_container_width=True)

        # CSV Download
        csv_issue = df_sel.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download Phrases for {view_label} (CSV)",
            csv_issue,
            f"issue_phrases_{selected_cat.lower()}.csv",
            "text/csv"
        )

# ----------------------------------------------------
# 5. MODEL EVALUATION
# ----------------------------------------------------
elif nav_choice == "Model Evaluation":
    st.markdown('<div class="section-title">Language Model Evaluation & Perplexity Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Evaluation of Unigram, Bigram, and Trigram language models on 350 held-out AppTek transcripts.</div>', unsafe_allow_html=True)

    if not artifacts or "perplexity" not in artifacts:
        st.error("Perplexity report not found. Run `python src/pipeline.py` first.")
    else:
        df_pp = artifacts["perplexity"]

        # Protocol Overview
        st.markdown("""
        <div class="academic-box">
            <b>Experimental Protocol:</b><br/>
            • <b>Partitioning:</b> Transcript-level split (80% Train, N=1,396 / 20% Held-Out Test, N=350, Random Seed 42).<br/>
            • <b>Zero Leakage:</b> Dialogue transcripts were partitioned whole; no utterances from test calls cross into training.<br/>
            • <b>Smoothing:</b> Laplace (Add-1) smoothing with fixed training vocabulary |V| = 16,186 (effective |V| = 16,187 with <code>&lt;/s&gt;</code>).<br/>
            • <b>Evaluation Metric:</b> Test-set cross-entropy sequence perplexity: PP(W) = exp( - 1/M &Sigma; ln P(w_i | context) ).
        </div>
        """, unsafe_allow_html=True)

        # Quantitative Results Table & Plot
        col_tbl, col_plt = st.columns([3, 2])
        with col_tbl:
            st.markdown("### Quantitative Perplexity Summary")
            st.dataframe(df_pp, use_container_width=True, height=240)

        with col_plt:
            fig_p = px.bar(
                df_pp,
                x="model",
                y="mean_perplexity",
                text="mean_perplexity",
                color="model",
                color_discrete_sequence=["#1E3A8A", "#2563EB", "#D97706"],
                title="Mean Perplexity across 350 Test Transcripts"
            )
            fig_p.update_traces(textposition="outside")
            fig_p.update_layout(height=280, margin=dict(l=10, r=20, t=30, b=20))
            st.plotly_chart(fig_p, use_container_width=True)

        # Mathematical Justification
        st.markdown('<div class="section-title">Mathematical & Empirical Analysis for Academic Viva</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="academic-box">
            <b>Why is Trigram Perplexity (2,845.61) higher than Unigram (551.93) and Bigram (586.42)?</b><br/><br/>
            1. <b>Massive Parameter Space Sparsity:</b><br/>
            With an effective vocabulary of |V| = 16,187, the theoretical Trigram context space contains |V|² &approx; 2.62 &times; 10⁸ possible state combinations.
            In unconstrained, multi-domain conversational speech across 16 domains and 14 accents, the vast majority of 3-word combinations in the test transcripts
            were never observed during training.<br/><br/>
            2. <b>Severe Add-1 Smoothing Discounting:</b><br/>
            When an unseen context (w_{i-2}, w_{i-1}) occurs in a test sentence, Laplace Add-1 smoothing assigns:
            $$P(w_i \\mid w_{i-2}, w_{i-1}) = \\frac{0 + 1}{0 + 16,187} \\approx 6.18 \\times 10^{-5}$$
            Accumulating hundreds of &minus;ln(6.18 &times; 10⁻⁵) &approx; 9.69 penalties heavily depresses sequence probability, leading to elevated exponential perplexity.<br/><br/>
            3. <b>Academic Defense Note:</b><br/>
            This empirical outcome is completely authentic and standard in statistical language modeling.
            Laplace smoothing successfully guarantees non-zero probabilities and prevents infinite perplexity crashes, but in industrial settings higher-order models
            rely on advanced backoff discounting algorithms (such as Good-Turing or Kneser-Ney).
        </div>
        """, unsafe_allow_html=True)

        # PDF Download Section
        st.markdown('<div class="section-title">Official Academic Reports & Documentation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Download publication-ready documentation generated directly from verified experimental pipeline metrics.</div>', unsafe_allow_html=True)

        d_col1, d_col2 = st.columns(2)
        arch_pdf = BASE_DIR / "docs" / "Architecture_and_Methodology.pdf"
        eval_pdf = BASE_DIR / "docs" / "Evaluation_Report.pdf"

        with d_col1:
            if arch_pdf.exists():
                with open(arch_pdf, "rb") as f:
                    st.download_button(
                        "📄 Download Architecture & Methodology PDF",
                        f.read(),
                        "CALLNGRAM_Architecture_and_Methodology.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Architecture PDF not found. Run `python src/generate_reports.py`.")

        with d_col2:
            if eval_pdf.exists():
                with open(eval_pdf, "rb") as f:
                    st.download_button(
                        "📊 Download Experimental Evaluation Report PDF",
                        f.read(),
                        "CALLNGRAM_Evaluation_Report.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Evaluation Report PDF not found. Run `python src/generate_reports.py`.")
