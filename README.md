# CALLNGRAM: Customer Call Language Analytics via N-gram Language Modeling

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-N--gram%20Language%20Modeling-green.svg)]()
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A B.Tech AIML Mini-Project in Natural Language Processing & Statistical Language Modeling**

---

## 1. Problem Statement
Customer call centers handle thousands of spoken interactions daily across telecommunications, banking, insurance, and retail domains. Automated quality assurance, agent assist, and query routing require rigorous quantification of spoken dialogue predictability and the discovery of recurring multi-word phrases. While deep neural models offer high capacity, their computational overhead and opaque probability structures make classical statistical N-gram language models essential for transparent, computationally lightweight, and mathematically grounded audio dialogue analytics.

---

## 2. Objective
**CALLNGRAM** builds an end-to-end NLP system for call-center dialogue analytics:
1. Formulate and implement **Unigram ($N=1$)**, **Bigram ($N=2$)**, and **Trigram ($N=3$)** statistical language models.
2. Implement **Laplace (Add-1) smoothing** for unconditional and conditional distributions to guarantee non-zero probabilities.
3. Compute intrinsic sequence predictability via **transcript Perplexity ($PP$)** on held-out test transcripts.
4. Mine recurrent customer issue phrases categorized by business domain and intent.
5. Deliver a professional, interactive **Streamlit web application** supporting live transcript scoring and data exploration.
6. Generate official academic documentation artifacts (**Architecture & Methodology PDF** and **Evaluation Report PDF**).

---

## 3. Dataset Description
The project employs two distinct corpora, strictly separated by structural purpose:

### Primary Corpus — AppTek Call-Center Dialogues
* **Nature**: Conversational call-center-style transcripts with agent and customer roles across multiple service domains.
* **Volume**: 1,746 transcripts (873 customer, 873 agent).
* **Domain Diversity**: 16 service domains (Banking, Technology, Delivery, Telecom, Travel, Agriculture, Health, etc.).
* **Accent Diversity**: 14 accent groups (en-US, en-GB, en-AU, en-IN, en-ZA, en-IE, etc.).
* **Average Length**: ~730.1 words per transcript (~1.3M total words).
* **Role in Project**: Core language modeling, Add-1 smoothing, held-out test perplexity evaluation.

### Secondary Corpus — Bitext Customer Support Dataset
* **Nature**: Curated customer service queries labeled by intent and category.
* **Volume**: 26,872 examples.
* **Taxonomy**: 11 business categories and 27 customer intents.
* **Average Length**: ~12 words per utterance.
* **Role in Project**: Customer issue phrase discovery and category/intent terminology profiling.

*Note: Neither dataset represents real wiretapped customer recordings. They are kept strictly separate to prevent domain distortion.*

---

## 4. Preprocessing & Spoken Marker Handling
Spoken conversational transcripts differ substantially from formal written text. To model customer service dialogue faithfully:
- **Case Normalization**: All text is converted to lowercase.
- **Whitespace Normalization**: Multiple spaces, tabs, and line breaks are normalized.
- **Discourse Marker Preservation**: Natural spoken hesitation markers (*uh*, *um*, *hmm*, *okay*, *yeah*, *mhm*) are intentionally preserved to accurately capture spoken dialogue dynamics.
- **Sentence Segmentation**: Sentences are isolated using NLTK `sent_tokenize` with a regex-based fallback for zero-dependency offline environments.
- **Boundary Handling**: Sentence boundary tokens (`<s>` and `</s>`) are incorporated for conditional sequence modeling.

---

## 5. Statistical Language Modeling & Laplace Smoothing

### Unigram Model:
$$P_{\text{Laplace}}(w) = \frac{\text{count}(w) + 1}{N + |V|}$$

### Conditional Bigram Model:
$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + |V|}$$

### Conditional Trigram Model:
$$P_{\text{Laplace}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i) + 1}{\text{count}(w_{i-2}, w_{i-1}) + |V|}$$

Where $N$ is total training tokens ($N = 1,056,726$), and $|V|$ is the fixed vocabulary size ($|V| = 16,186$ unique words observed during training).

---

## 6. Perplexity Evaluation
Perplexity evaluates the predictive fit of the trained reference model over an unseen sequence of $M$ tokens:
$$PP(W) = \exp\left( -\frac{1}{M} \sum_{i=1}^M \ln P(w_i \mid \text{context}) \right)$$

### Benchmark Results on 350 Held-Out AppTek Test Transcripts (80/20 Split, Seed=42):
| Language Model | Mean Perplexity | Median Perplexity | Transcripts Evaluated |
| :--- | :---: | :---: | :---: |
| **Unigram (Add-1)** | **551.93** | **497.73** | 350 |
| **Bigram (Add-1)** | **586.42** | **520.03** | 350 |
| **Trigram (Add-1)** | **2,845.61** | **2,620.89** | 350 |

*Data source: `data/annotated/perplexity_report.csv` generated by `src/pipeline.py`.*

---

## 7. Customer Issue Intelligence
From the 26,872 Bitext customer utterances, recurring multi-word collocations were identified across:
- **11 Categories**: `ORDER`, `REFUND`, `PAYMENT`, `DELIVERY`, `ACCOUNT`, `CANCEL`, `INVOICE`, `SHIPPING`, `SUBSCRIPTION`, `FEEDBACK`, `CONTACT`.
- **27 Intents**: `cancel_order`, `get_refund`, `track_order`, `change_shipping_address`, `payment_issue`, etc.

*Important distinction: High frequency indicates recurring domain terminology, NOT automatic intent classification.*

---

## 8. Project Architecture
```text
CALLNGRAM/
├── app.py                         # Streamlit Interactive Web Application
├── requirements.txt               # Dependencies (Pandas, NLTK, Scikit-learn, Streamlit, ReportLab)
├── runtime.txt                    # Python runtime specification (python-3.12)
├── README.md                      # Project Documentation
├── FINAL_PROJECT_CHECKLIST.md     # Verified Status Checklist
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing.py           # Text cleaning, spoken markers, segmentation
│   ├── models.py                  # Unigram, Bigram, Trigram, Laplace, Perplexity
│   ├── pipeline.py                # End-to-end execution pipeline
│   └── generate_reports.py        # ReportLab PDF documentation generator
│
├── data/
│   ├── raw_transcripts/           # Raw AppTek and Bitext source datasets
│   └── annotated/                 # Precomputed CSV reports and project summary JSON
│       ├── apptek_clean.csv
│       ├── bitext_clean.csv
│       ├── perplexity_report.csv
│       ├── apptek_1gram_frequency.csv
│       ├── apptek_2gram_frequency.csv
│       ├── apptek_3gram_frequency.csv
│       ├── 1gram_probability_report.csv
│       ├── 2gram_probability_report.csv
│       ├── 3gram_probability_report.csv
│       ├── bitext_issue_phrase_report.csv
│       └── project_summary.json
│
├── reports/                       # Exported evaluation summaries and PDF reports
│   ├── Architecture_and_Methodology.pdf
│   └── Evaluation_Report.pdf
│
├── docs/                          # Official documentation and report PDFs
│   ├── Architecture_and_Methodology.pdf
│   ├── Architecture_and_Methodology.md
│   ├── Evaluation_Report.pdf
│   └── Evaluation_Report.md
│
└── tests/
    └── test_callngram.py          # Automated Test Suite (11 Tests)
```

---

## 9. Installation & Running Locally

### Step 1: Clone or Navigate to Project
```bash
git clone https://github.com/your-username/CALLNGRAM.git
cd CALLNGRAM
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Full Pipeline & Generate Reports (Optional)
```bash
python src/pipeline.py
```

### Step 4: Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 5: Run Automated Tests
```bash
python -m pytest tests/
```

---

## 10. Deployment Instructions

### Streamlit Community Cloud (Recommended)
1. Initialize and push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Deploy CALLNGRAM: Customer Call Language Analytics"
   git push origin main
   ```
2. Log into [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **"New app"** and select your GitHub repository.
4. Set Main file path: `app.py`.
5. Click **"Deploy!"**.
6. The application loads instantly using precomputed artifacts and cached models.

---

## 11. Academic Limitations & Observations
1. **Laplace Sparsity Penalty**: Add-1 smoothing allocates uniform probability to unseen transitions, creating a heavy cross-entropy penalty for high-order trigrams ($|V|^2 \approx 2.6 \times 10^8$ context states).
2. **Context Horizon**: N-gram models are limited to $N-1$ preceding tokens and do not capture multi-turn conversational discourse state.
3. **Conversational vs Written**: Perplexity benchmarks are inherently higher on spontaneous conversational dialogue than on formal written news text.

---

## 12. Student / Author Details
- **Project Title**: CALLNGRAM — Customer Call Language Analytics via N-gram Language Modeling
- **Course**: B.Tech Artificial Intelligence and Machine Learning (AIML)
- **Subject**: Mini-Project / Natural Language Processing
