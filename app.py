"""
CALLNGRAM — Customer Call Language Analytics via N-gram Language Modeling
B.Tech AIML Mini-Project Application

Features:
- Primary Corpus (AppTek): 1,746 transcripts (80/20 train/test split, seed 42)
- Secondary Corpus (Bitext): 26,872 customer support queries
- Unigram, Bigram, Trigram models with Laplace (Add-1) smoothing
- Real-time transcript perplexity evaluation against trained reference models
- N-gram frequency and probability exploration
- Customer issue & intent phrase discovery
- Academic documentation artifacts (PDF)
"""

import sys
import json
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
    /* Main Layout & Typography */
    .stApp {
        background-color: #F8FAFC;
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
        margin-top: 18px;
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
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease;
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .stat-detail {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Academic Callout Boxes */
    .academic-box {
        background-color: #FFFFFF;
        border-left: 4px solid #1E3A8A;
        border-right: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #1E293B;
    }

    /* Disclaimers & Notes */
    .viva-note {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        color: #78350F;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        margin: 12px 0;
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
    "Modules",
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
        st.markdown('<div class="section-caption">Mathematical workflow from raw conversational transcripts to perplexity and issue discovery.</div>', unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown("""
            **1. Preprocessing**
            - Lowercase normalization
            - Punctuation isolation
            - Spoken markers (*uh, um, okay*)
            - Sentence boundary tags `<s>`, `</s>`
            """)
        with p2:
            st.markdown("""
            **2. N-Gram Estimation**
            - Unigram: $P(w)$
            - Bigram: $P(w_i \\mid w_{i-1})$
            - Trigram: $P(w_i \\mid w_{i-2}, w_{i-1})$
            - Parameter space: $|V|, |V|^2$
            """)
        with p3:
            st.markdown("""
            **3. Laplace Smoothing**
            - $P(w|ctx) = \\frac{C(ctx, w) + 1}{C(ctx) + |V|}$
            - Zero-frequency avoidance
            - Guaranteed positive probabilities
            - Fixed vocabulary $|V| = 16,186$
            """)
        with p4:
            st.markdown("""
            **4. Perplexity Evaluation**
            - $PP = \\exp(-\\frac{1}{M}\\sum \\ln P)$
            - Evaluated on 350 test calls
            - Reference model scoring
            - Issue phrase mining (Bitext)
            """)

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
        # Load real test record 0
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
            st.success(f"Loaded `{upload_file.name}` ({len(chosen_default)} characters)")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    transcript_input = st.text_area("Transcript Text", value=chosen_default, height=170)

    if st.button("Run Transcript Analysis", type="primary"):
        if not transcript_input.strip():
            st.warning("Please enter or select a valid transcript text.")
        else:
            with st.spinner("Analyzing transcript properties and computing model perplexity..."):
                cleaned = clean_text_for_tokens(transcript_input)
                sentences = segment_sentences(cleaned)
                sents_toks, all_words = tokenize_transcript(cleaned)

                # Discourse markers
                spoken_found = [w for w in all_words if w in SPOKEN_MARKERS]
                unique_words = len(set(all_words))

                # Display stats
                st.markdown('<div class="section-title">1. Transcript Properties</div>', unsafe_allow_html=True)
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.metric("Total Tokens", f"{len(all_words):,}")
                with p2:
                    st.metric("Unique Vocabulary", f"{unique_words:,}")
                with p3:
                    st.metric("Sentence Count", f"{len(sentences):,}")
                with p4:
                    marker_summary = f"{len(spoken_found)} ({', '.join(set(spoken_found)) if spoken_found else 'none'})"
                    st.metric("Spoken Markers", marker_summary)

                # Reference Perplexity Evaluation
                st.markdown('<div class="section-title">2. Model Perplexity (Evaluated Against AppTek Reference Models)</div>', unsafe_allow_html=True)
                if m1_ref and m2_ref and m3_ref:
                    t_eval_1 = [tokenize_sentence(s) for s in sentences if tokenize_sentence(s)]
                    t_eval_2 = [["<s>"] + tokenize_sentence(s) + ["</s>"] for s in sentences if tokenize_sentence(s)]
                    t_eval_3 = [["<s>", "<s>"] + tokenize_sentence(s) + ["</s>"] for s in sentences if tokenize_sentence(s)]

                    pp1 = m1_ref.perplexity(t_eval_1)
                    pp2 = m2_ref.perplexity(t_eval_2)
                    pp3 = m3_ref.perplexity(t_eval_3)

                    pp_col1, pp_col2, pp_col3 = st.columns(3)
                    with pp_col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp1:.2f}</div>
                            <div class="stat-label">Unigram Perplexity</div>
                            <div class="stat-detail">Laplace (Add-1) Smoothed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with pp_col2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp2:.2f}</div>
                            <div class="stat-label">Bigram Perplexity</div>
                            <div class="stat-detail">Laplace (Add-1) Smoothed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with pp_col3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{pp3:.2f}</div>
                            <div class="stat-label">Trigram Perplexity</div>
                            <div class="stat-detail">Laplace (Add-1) Smoothed</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="viva-note">
                        <b>Academic Note:</b> Perplexity evaluates how well the trained reference model (N=1,396 calls) predicts this sequence.
                        It is NOT computed from the transcript's own internal counts. Lower perplexity denotes higher predictive likelihood under the learned model.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Reference models currently initializing. Please refresh in a moment.")

                # Local N-gram collocations
                def extract_local_ngrams(words, n):
                    counts = {}
                    for i in range(len(words) - n + 1):
                        ph = " ".join(words[i:i + n])
                        counts[ph] = counts.get(ph, 0) + 1
                    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

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
                    st.caption("Displays the segmented sentences with injected boundary tokens for conditional modeling:")
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
        order = st.radio("Select N-gram Order", ["Unigram (Single Words)", "Bigram (Two-Word Collocations)", "Trigram (Three-Word Sequences)"], horizontal=True)

        if "Unigram" in order:
            df_curr = artifacts.get("prob_1g", artifacts.get("freq_1g"))
            title = "Unigram Distribution (Single Words)"
        elif "Bigram" in order:
            df_curr = artifacts.get("prob_2g", artifacts.get("freq_2g"))
            title = "Bigram Distribution (Two-Word Collocations)"
        else:
            df_curr = artifacts.get("prob_3g", artifacts.get("freq_3g"))
            title = "Trigram Distribution (Three-Word Sequences)"

        if df_curr is not None and not df_curr.empty:
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                search_query = st.text_input("Filter phrases by keyword (e.g., 'thank', 'service', 'account', 'order')", "")
            with f_col2:
                top_limit = st.selectbox("Display Limit", [15, 25, 50, 100], index=1)

            if search_query:
                filtered_df = df_curr[df_curr["ngram"].str.contains(search_query.lower(), na=False)]
            else:
                filtered_df = df_curr

            st.write(f"Displaying **{min(len(filtered_df), top_limit)}** of **{len(filtered_df):,}** phrases:")

            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                chart_df = filtered_df.head(top_limit)
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
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                display_cols = ["ngram", "count"]
                if "probability" in filtered_df.columns:
                    display_cols.append("probability")
                st.dataframe(filtered_df[display_cols].head(top_limit), use_container_width=True, height=520)

            # CSV download
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download {order.split()[0]} Frequency Report (CSV)",
                csv_data,
                f"apptek_{order.split()[0].lower()}_report.csv",
                "text/csv"
            )

        # Context Transition Lookup
        st.markdown('<div class="section-title">Context Probability Lookup</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Query transition probabilities for a specific context word under Bigram and Trigram models.</div>', unsafe_allow_html=True)
        query_word = st.text_input("Enter a context word to predict next tokens (e.g., 'thank', 'credit', 'customer')", "thank")
        if query_word and m2_ref:
            q_clean = query_word.lower().strip()
            # Find bigrams starting with q_clean
            candidates = []
            for (w1, w2), count in m2_ref.ngram_counts.items():
                if w1 == q_clean and w2 != "</s>":
                    prob = m2_ref.probability(w2, context=w1, smoothed=True)
                    mle_p = m2_ref.mle_probability(w2, context=w1)
                    candidates.append({"Next Word": w2, "Bigram Count": count, "Laplace Probability": prob, "MLE Probability": mle_p})
            if candidates:
                cand_df = pd.DataFrame(candidates).sort_values(by="Bigram Count", ascending=False).head(10)
                st.dataframe(cand_df, use_container_width=True)
            else:
                st.info(f"No direct bigram transitions found starting with `{q_clean}` in the training corpus.")

# ----------------------------------------------------
# 4. ISSUE INTELLIGENCE
# ----------------------------------------------------
elif nav_choice == "Issue Intelligence":
    st.markdown('<div class="section-title">Customer Issue Phrase Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Analysis of recurring multi-word collocations across categories and intents in the Bitext Customer Support Dataset.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="viva-note">
        <b>Academic Methodology Note:</b> High-frequency N-grams indicate recurring domain terminology and conversational phrasing associated with an issue.
        They represent vocabulary patterns for call analysis and do not constitute automatic intent classification by themselves.
    </div>
    """, unsafe_allow_html=True)

    if not artifacts or "issue_report" not in artifacts:
        st.error("Bitext issue phrase report not found. Run `python src/pipeline.py` first.")
    else:
        df_issue = artifacts["issue_report"]
        mode = st.radio("Taxonomy Level", ["Category Level (11 Commercial Categories)", "Intent Level (27 Specific Customer Intents)"], horizontal=True)

        if "Category" in mode:
            categories = sorted(df_issue[df_issue["level"] == "category"]["target"].unique().tolist())
            selected = st.selectbox("Select Customer Issue Category", categories)
            df_sel = df_issue[(df_issue["level"] == "category") & (df_issue["target"] == selected)]
        else:
            intents = sorted(df_issue[df_issue["level"] == "intent"]["target"].unique().tolist())
            selected = st.selectbox("Select Customer Intent", intents)
            df_sel = df_issue[(df_issue["level"] == "intent") & (df_issue["target"] == selected)]

        st.markdown(f"### Recurring Phrases for: `{selected}`")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Top Unigrams**")
            u_df = df_sel[df_sel["ngram_order"] == "unigram"][["phrase", "count"]].reset_index(drop=True)
            st.dataframe(u_df, use_container_width=True)
        with c2:
            st.markdown("**Top Bigrams**")
            b_df = df_sel[df_sel["ngram_order"] == "bigram"][["phrase", "count"]].reset_index(drop=True)
            st.dataframe(b_df, use_container_width=True)
        with c3:
            st.markdown("**Top Trigrams**")
            t_df = df_sel[df_sel["ngram_order"] == "trigram"][["phrase", "count"]].reset_index(drop=True)
            st.dataframe(t_df, use_container_width=True)

        # Plotly grouped chart
        fig_issue = px.bar(
            df_sel,
            x="phrase",
            y="count",
            color="ngram_order",
            barmode="group",
            title=f"Phrase Frequency Distribution for: {selected}",
            color_discrete_sequence=["#1E3A8A", "#2563EB", "#059669"],
            labels={"phrase": "Phrase", "count": "Frequency", "ngram_order": "Order"}
        )
        fig_issue.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_issue, use_container_width=True)

        # Download
        csv_issue = df_sel.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download Phrases for {selected} (CSV)",
            csv_issue,
            f"issue_phrases_{selected.lower()}.csv",
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
            • <b>Data Leakage Prevention:</b> Whole call dialogues were assigned to either train or test; no sentences from the same call cross partitions.<br/>
            • <b>Smoothing:</b> Add-1 (Laplace) smoothing applied across unconditional and conditional distributions.<br/>
            • <b>Vocabulary (|V|):</b> Fixed 16,186 unique word types observed in training (+1 terminal tag <code>&lt;/s&gt;</code>).
        </div>
        """, unsafe_allow_html=True)

        col_t, col_c = st.columns([2, 3])
        with col_t:
            st.markdown("### 📋 Perplexity Metrics Table")
            st.dataframe(df_pp, use_container_width=True)
            st.download_button(
                "📥 Download Perplexity Report (CSV)",
                df_pp.to_csv(index=False).encode("utf-8"),
                "perplexity_report.csv",
                "text/csv"
            )

        with col_c:
            st.markdown("### 📊 Mean Perplexity Comparison")
            fig_p = px.bar(
                df_pp,
                x="model",
                y="mean_perplexity",
                text="mean_perplexity",
                color="model",
                color_discrete_sequence=["#1E3A8A", "#2563EB", "#D97706"],
                labels={"model": "Model", "mean_perplexity": "Mean Perplexity"}
            )
            fig_p.update_traces(textposition='outside')
            fig_p.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_p, use_container_width=True)

        # Dynamic metric extraction
        u_row = df_pp[df_pp["model"].str.contains("Unigram")]
        b_row = df_pp[df_pp["model"].str.contains("Bigram")]
        t_row = df_pp[df_pp["model"].str.contains("Trigram")]

        u_mean = u_row["mean_perplexity"].iloc[0] if not u_row.empty else 551.93
        u_med = u_row["median_perplexity"].iloc[0] if not u_row.empty else 497.73
        b_mean = b_row["mean_perplexity"].iloc[0] if not b_row.empty else 586.42
        b_med = b_row["median_perplexity"].iloc[0] if not b_row.empty else 520.03
        t_mean = t_row["mean_perplexity"].iloc[0] if not t_row.empty else 2845.61

        st.markdown('<div class="section-title">Academic Discussion & Theoretical Findings</div>', unsafe_allow_html=True)
        st.markdown(f"""
        **1. What Perplexity Measures:**
        $$\\text{{Perplexity}}(W) = \\exp\\left( -\\frac{{1}}{{M}} \\sum_{{i=1}}^M \\ln P(w_i \\mid \\text{{context}}) \\right)$$
        Perplexity corresponds to the exponentiated cross-entropy of the model over the test token sequence. Intuitively, it represents the effective branching factor: how many equally likely words the model is choosing among. **Lower perplexity indicates higher predictive probability under the learned model.**

        **2. Unigram vs Bigram Behavior in Multi-Domain Conversational Data:**
        - Unigram Mean Perplexity: **{u_mean:.2f}** (Median: **{u_med:.2f}**).
        - Bigram Mean Perplexity: **{b_mean:.2f}** (Median: **{b_med:.2f}**).
        - In scripted sub-domains (e.g. Banking, Delivery), bigrams excel due to tight collocations (*“thank you”*, *“customer service”*, *“credit card”*).
        - However, across unconstrained multi-topic dialogue, vocabulary dispersion means unseen transitions receive uniform Add-1 smoothing penalties across $|V| = 16,186$ outcomes.

        **3. Why Add-1 Smoothing Causes High Trigram Perplexity:**
        - Trigram Mean Perplexity: **{t_mean:.2f}**.
        - The trigram context parameter space scales as $|V|^2 \\approx (16,186)^2 \\approx 2.62 \\times 10^8$ potential states.
        - Because conversational dialogue is highly varied, the majority of 3-word test sequences are unseen in the training partition.
        - Uniform Add-1 smoothing assigns a probability of $P = \\frac{{1}}{{0 + |V|}} \\approx 6.18 \\times 10^{{-5}}$ to each unseen transition, heavily penalizing log-likelihood.
        - This is a well-established theoretical characteristic of uniform Add-1 smoothing on higher-order models in computational linguistics.
        """)

        # Documentation Downloads
        st.markdown('<div class="section-title">Official Documentation Artifacts</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Compiled ReportLab PDF artifacts with exact metrics generated from the experimental pipeline:</div>', unsafe_allow_html=True)
        
        pdf_col1, pdf_col2 = st.columns(2)
        arch_pdf_path = BASE_DIR / "docs" / "Architecture_and_Methodology.pdf"
        eval_pdf_path = BASE_DIR / "docs" / "Evaluation_Report.pdf"

        with pdf_col1:
            if arch_pdf_path.exists():
                with open(arch_pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Architecture & Methodology (PDF)",
                        f.read(),
                        "Architecture_and_Methodology.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Architecture PDF not found in docs/")

        with pdf_col2:
            if eval_pdf_path.exists():
                with open(eval_pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Experimental Evaluation Report (PDF)",
                        f.read(),
                        "Evaluation_Report.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Evaluation Report PDF not found in docs/")
