# CALLNGRAM — FINAL PROJECT CHECKLIST

Each item is strictly audited against actual code execution and test verification:
- `[x]` Verified (tested and confirmed working)
- `[ ]` Pending (requires external / user authentication step)
- `[!]` Problem / needs attention

---

### 1. Dataset
- [x] **AppTek loaded**: 1,746 call-center-style conversational transcripts in `data/raw_transcripts/apptek/`
- [x] **Bitext loaded**: 26,872 customer support utterances in `data/raw_transcripts/`
- [x] **Dataset statistics verified**: 16 domains, 14 accents, 873 customer and 873 agent dialogues; 11 Bitext categories, 27 intents
- [x] **No accidental dataset merge**: Kept strictly as separate corpora; AppTek for language modeling, Bitext for issue/intent analysis
- [x] **Wording verified**: No claims of "real wiretapped customer calls" or "genuine phone audio"; accurately described as conversational benchmark transcripts

### 2. Preprocessing
- [x] **Tokenization**: Handled with punctuation stripping while preserving internal apostrophes
- [x] **Sentence segmentation**: Robust `sent_tokenize` with automatic regex fallback for offline environments
- [x] **Normalization**: Lowercase and whitespace normalization applied uniformly
- [x] **Spoken marker handling**: Conversation markers (*uh, um, hmm, okay, yeah, mhm*) intentionally preserved
- [x] **Boundary tokens**: `<s>` and `</s>` injected for conditional probability transitions

### 3. Language Modeling
- [x] **Unigram Language Model**: $P(w) = \frac{\text{count}(w) + 1}{N + |V|}$
- [x] **Bigram Language Model**: $P(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + |V|}$
- [x] **Trigram Language Model**: $P(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i) + 1}{\text{count}(w_{i-2}, w_{i-1}) + |V|}$
- [x] **Laplace (Add-1) smoothing**: Guaranteed non-zero probabilities for all transitions
- [x] **Consistent vocabulary**: Fixed $|V| = 16,186$ across training, probability tables, and evaluation
- [x] **Perplexity computation**: $PP(W) = \exp\left(-\frac{1}{M} \sum \ln P\right)$ evaluated against reference model

### 4. Evaluation
- [x] **80/20 train/test split**: 1,396 training transcripts / 350 test transcripts with fixed seed 42
- [x] **No transcript leakage**: Transcript-level split; no sentences from the same call dialogue cross boundaries
- [x] **Complete training partition**: Full 1,396 transcripts utilized in reference models without arbitrary truncation
- [x] **Test evaluation**: Evaluated across all 350 held-out transcripts
- [x] **Mean/median perplexity recorded**:
  - Unigram: Mean 551.93 | Median 497.73
  - Bigram: Mean 586.42 | Median 520.03
  - Trigram: Mean 2,845.61 | Median 2,620.89

### 5. Analysis
- [x] **Frequency distributions**: Generated actual frequencies (`apptek_{1,2,3}gram_frequency.csv`)
- [x] **Domain analysis**: 16 domains identified and presented in dashboard
- [x] **Issue/category analysis**: Category-wise Unigram, Bigram, Trigram mining across 11 categories
- [x] **Intent analysis**: Intent-wise phrase mining across 27 intents (1,140 rows in `bitext_issue_phrase_report.csv`)
- [x] **Distinction noted**: High frequency clearly documented as recurring domain terminology, not automatic intent classification

### 6. Interactive Application
- [x] **Streamlit dashboard**: Clean, professional single-file `app.py`
- [x] **Transcript analysis**: Live text input and TXT file upload evaluated against trained reference model
- [x] **N-gram explorer**: Unigram/Bigram/Trigram interactive exploration with search filter, charts, and CSV downloads
- [x] **Issue intelligence**: Interactive category and intent drill-downs
- [x] **Model evaluation view**: Academic metrics table, comparison bar chart, theoretical explanation, and PDF download buttons

### 7. Documentation
- [x] **README.md**: Comprehensive, technical, with clear setup, run, and evaluation commands
- [x] **Architecture and Methodology PDF**: `docs/Architecture_and_Methodology.pdf` generated via ReportLab
- [x] **Evaluation Report PDF**: `docs/Evaluation_Report.pdf` generated via ReportLab
- [x] **Reports mirrored**: Copies saved in `reports/` folder
- [x] **Formulas and architecture diagrams**: Formally documented in markdown and PDF

### 8. Testing
- [x] **Automated tests created**: `tests/test_callngram.py` with 11 distinct test cases
- [x] **All tests actually passed**: 11/11 passed in automated test runner (Pytest / Unittest)

### 9. Clean Architecture
- [x] **Nested duplicates removed**: `CALLNGRAM/` directory deleted; single clean workspace root
- [x] **Unneeded web artifacts removed**: React, Vite, Tailwind, TypeScript, Express, and Gemini SDK files deleted
- [x] **No hardcoded local paths**: Only relative paths used across entire Python codebase

### 10. Deployment
- [x] **requirements.txt**: Fully verified with exact dependencies
- [x] **runtime.txt**: Configured with `python-3.12`
- [x] **GitHub Repository Configured & Pushed**: Verified live on `https://github.com/mvparekh/CALLINGRAM` on branch `main`
- [x] **Streamlit Community Cloud-ready**: App entry point `app.py` validated and tested locally
- [ ] **Live Public URL**: Requires 1-click authorization on share.streamlit.io (Select repository `mvparekh/CALLINGRAM`, file `app.py`, click Deploy)
