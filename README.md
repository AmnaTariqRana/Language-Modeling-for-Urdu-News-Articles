# Neural NLP Pipeline: From Embeddings to Transformers


This repository contains a comprehensive, end-to-end Natural Language Processing pipeline built entirely from scratch using **PyTorch**. The project transitions from foundational word vectorization to advanced sequence labeling with BiLSTMs and structured topic classification using a custom-built Transformer Encoder.

---

##  Project Overview

The pipeline is divided into three core modules, focusing on the processing and understanding of **BBC Urdu** news data.

### 1. Word Embeddings (Lexical Semantics)

Implemented and compared multiple vectorization techniques to capture semantic relationships:

* **TF-IDF & PPMI Baselines:** Built term-document and co-occurrence matrices (k=5).


* **Skip-gram Word2Vec:** Implemented from scratch with **Negative Sampling** (K=10) and noise distribution.


* **Evaluation:** Conducted analogy tests (a:b::c:?) and cosine similarity analysis for query words like *Pakistan, Adalat,* and *Fauj*.



### 2. Sequence Labeling (POS & NER)

Developed a multi-task learning framework for Urdu linguistics:

* **Architecture:** 2-layer **Bidirectional LSTM** with Dropout (p=0.5).


* **POS Tagging:** 12-tag classification using a rule-based seed tagger and linear decoder.


* **Named Entity Recognition (NER):** Implemented a **Conditional Random Field (CRF)** layer with **Viterbi decoding** for structured entity detection (PER, LOC, ORG, MISC).


* **Ablation Study:** Analyzed the impact of unidirectional context, dropout regularization, and pre-trained vs. random embedding initialization.



### 3. Transformer Encoder (Topic Classification)

Engineered a custom Transformer architecture (no `nn.Transformer` built-ins) for 5-class news classification.

* **Multi-Head Attention:** 4 heads with scaled dot-product attention and padding masks.


* **Positional Encoding:** Fixed sinusoidal signals added to input embeddings.


* **Optimization:** Utilized **AdamW** with a **Cosine Learning Rate Schedule** and warmup steps.


* **Interpretability:** Generated attention heatmaps to visualize token-level importance during classification.



---

##  Tech Stack

* **Language:** Python
* **Framework:** PyTorch (Core Tensor Ops, Autograd) 


* **Visualization:** Matplotlib, Seaborn (t-SNE plots, attention heatmaps) 


* **Data Handling:** NumPy, JSON 






---

##  Key Results & Insights

* **Embeddings:** Skip-gram (Condition C3) outperformed PPMI baselines in capturing complex analogies.


* **CRF Benefit:** The addition of a CRF layer significantly improved NER F1-scores by enforcing valid tag transitions (e.g., B-PER followed by I-PER).


* **Attention Analysis:** Transformer heatmaps revealed high focus on indicative keywords (e.g., "election" for Politics, "GDP" for Economy).



---

##  Getting Started

1. **Clone the Repo:**
```bash
git clone https://github.com/AmnaTariqRana/code.git

```


2. **Install Dependencies:**
```bash
pip install torch numpy matplotlib scikit-learn

```


3. **Run the Notebook:**
Ensure `cleaned.txt`, `raw.txt`, and `Metadata.json` are in the root directory before executing the cells.



---
