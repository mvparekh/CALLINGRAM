# CALLNGRAM: Architecture and Methodology

## 1. System Overview

**CALLNGRAM** is an NLP system designed for **Customer Call Language Analytics via N-gram Language Modeling**.
The system analyzes spoken customer-service dialogues and customer queries to model language predictability, discover recurrent multi-word phrases, and identify issue-specific conversational markers.

```
+-----------------------------------------------------------------------------------+
|                               DATASET ARCHITECTURE                                |
|                                                                                   |
|   PRIMARY CORPUS: AppTek Call-Center Dialogues   SECONDARY CORPUS: Bitext 27K      |
|   • 1,746 Transcripts (873 Agent, 873 Customer) • 26,872 Customer Inquiries       |
|   • 16 Service Domains, 14 Global Accents        • 11 Categories, 27 Intents      |
|   • Conversational dialogue sequences           • Customer issue phrase mining    |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                PREPROCESSING                                      |
|   • Lowercase normalization                                                       |
|   • Punctuation filtering (preserving word structures)                           |
|   • Spoken discourse marker preservation (uh, um, hmm, okay, yeah, mhm)           |
|   • Sentence boundary segmentation (NLTK punkt)                                   |
|   • Boundary token injection: <s>, </s> for conditional modeling                  |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        N-GRAM STATISTICAL LANGUAGE MODELS                         |
|   • Unigram Model (N=1): Lexical frequency and probability                        |
|   • Bigram Model (N=2): Two-word transition distribution                          |
|   • Trigram Model (N=3): Three-word context-conditioned probability               |
|                                                                                   |
|   LAPLACE (ADD-1) SMOOTHING:                                                      |
|   P(w | context) = [count(context, w) + 1] / [count(context) + |V|]               |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                          EVALUATION & MINING PIPELINE                             |
|   • Transcript-Level Train/Test Split (80% Train N=1,396 / 20% Held-Out N=350)   |
|   • Perplexity Computation: PP = exp( - 1/N * sum(ln P(w_i | context)) )          |
|   • Frequency & Smoothed Probability Reports                                      |
|   • Category & Intent Phrase Association Mining (Bitext Intelligence)             |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            INTERACTIVE USER INTERFACE                             |
|   • Streamlit Web Application (app.py) & Cloud Deployment                         |
|   • Real-Time Transcript Upload & Perplexity Scoring                             |
|   • N-gram Interactive Distribution Visualizer                                    |
|   • Customer Issue Discovery Engine & CSV Data Exports                            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dataset Separation Rationale

A central architectural decision in CALLNGRAM is the strict separation between the **Primary Corpus** and the **Secondary Corpus**:

| Property | Primary: AppTek Call-Center Dialogues | Secondary: Bitext Customer Support Dataset |
| :--- | :--- | :--- |
| **Origin** | AppTek Benchmark Dialogues | Bitext Support Query Collection |
| **Records** | 1,746 transcripts (873 agent, 873 customer) | 26,872 customer queries |
| **Average Length** | ~756 words per transcript | ~12 words per utterance |
| **Structure** | Full multi-turn conversational transcripts | Short, targeted single-intent customer prompts |
| **Domain Coverage**| 16 distinct service domains, 14 accents | 11 categories, 27 customer intents |
| **Role in Project**| Language modeling, probability distribution, Laplace smoothing, transcript perplexity | Issue phrase discovery, category/intent phrase association |

**Why they are kept separate:**
1. **Structural Mismatch**: AppTek consists of extended conversations containing both caller and agent dialogue with conversational fillers. Bitext consists of concise customer questions or requests. Merging them into a single corpus would introduce severe domain distribution distortion and invalidate statistical perplexity benchmarks.
2. **Evaluation Integrity**: The test split for perplexity is drawn strictly from AppTek test transcripts, ensuring that cross-entropy measures continuous spoken dialogue rather than repetitive synthetic customer questions.

---

## 3. Preprocessing and Spoken Marker Handling

Spoken conversational transcripts differ substantially from formal written text. To model customer service dialogue faithfully:

### 3.1 Preservation of Conversational Spoken Markers
Conversational hesitation markers (*uh*, *um*, *hmm*, *okay*, *yeah*, *mhm*) are **not removed**. In customer call centers, these markers carry important linguistic information:
- Customer hesitation or uncertainty (*"uh, I don't know my account number"*)
- Agent confirmation and pacing (*"okay, um let me verify that"*)

### 3.2 Cleaning Pipeline
1. Standardizes Unicode quotation marks and apostrophes.
2. Normalizes bracketed notation `(uh)` $\to$ `uh` and `(um,)` $\to$ `um`.
3. Normalizes whitespace and strips transcription artifacts (e.g. interruption tildes `~`).
4. Performs sentence segmentation using NLTK `sent_tokenize`.
5. Cleans standalone punctuation while preserving word-internal apostrophes (`don't`, `i'm`).

---

## 4. Mathematical Methodology

### 4.1 N-Gram Probability Estimation
An N-gram model estimates the probability of a word given its preceding history using the Markov assumption:
$$P(w_1, w_2, \dots, w_N) = \prod_{i=1}^N P(w_i \mid w_{i-n+1}^{i-1})$$

### 4.2 Laplace (Add-1) Smoothing
To prevent zero-frequency probabilities on unseen word sequences, Add-1 smoothing redistributes probability mass evenly across the vocabulary $|V|$:

#### Unigram Model:
$$P_{\text{Laplace}}(w) = \frac{\text{count}(w) + 1}{N + |V|}$$
where $N$ is the total token count and $|V|$ is the unique vocabulary size.

#### Conditional Bigram Model:
$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + |V|}$$

#### Conditional Trigram Model:
$$P_{\text{Laplace}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i) + 1}{\text{count}(w_{i-2}, w_{i-1}) + |V|}$$

### 4.3 Perplexity Computation
Perplexity ($PP$) is the standard intrinsic evaluation metric for language models:
$$PP(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid \text{context}) \right)$$
- Intuitively, perplexity represents the **effective branching factor** of the language model.
- A lower perplexity indicates that the model assigns higher probability to the unseen test sequence (i.e. finding it more predictable).

---

## 5. Customer Issue Phrase Mining

The customer issue discovery component utilizes the structured category and intent annotations in the Bitext corpus:
1. Filters instructions by **Category** (e.g., `PAYMENT`, `ORDER`, `REFUND`, `ACCOUNT`, `DELIVERY`) or **Intent** (e.g., `cancel_order`, `get_refund`, `track_order`).
2. Extracts Unigrams, Bigrams, and Trigrams specific to each subset.
3. Quantifies phrase frequencies to uncover multi-word indicators (e.g., *"tracking number"*, *"credit card"*, *"cancel my subscription"*, *"billing statement"*).
4. Provides phrase association reports for root-cause analysis in customer support workflows.
