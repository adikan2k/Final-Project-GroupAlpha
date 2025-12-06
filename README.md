# Scholarly Topic Navigator — Explainable Research Digest Pipeline

Automated, explainable NLP system to reduce academic information overload for faculty labs.

---

## Overview

This project tackles the surge of NLP publications (arXiv, ACL, EMNLP, etc.) by building an automated pipeline that surfaces timely, relevant papers with transparent recommendation logic. It combines modern neural methods with standard NLP preprocessing to enable clustering, classification, retrieval, and summarization, while incorporating faculty feedback for continuous improvement.

---

## Repository Structure

```
Final-Project-Group-LexiCore/
├── code/                          # Jupyter notebooks with implementation
│   ├── Retrieval_Topic_Modeling.ipynb        # Core retrieval & topic modeling pipeline
│   ├── Week1_implementation_AdityaKanbargi.ipynb  # Week 1 data ingestion & preprocessing 
│   └── Week_3_Integration_UI_Explainability_Trisha.ipynb  # classification summarizatioUI & explainability integration
│
├── data/                          # Data files and artifacts
│   ├── raw/                       # Raw downloaded datasets (arXiv, etc.)
│   ├── processed/                 # Cleaned and preprocessed data
│   ├── Retrieval_Topic_Modeling/  # Data for retrieval & topic modeling
│   │   ├── data/                  # Processed datasets
│   │   ├── sample_data/           # Sample data for testing
│   │   └── visualizations/        # Generated visualization outputs
│   └── Trisha_Week_3/             # Week 3 implementation artifacts
│       ├── app.py                 # Streamlit/Gradio web app for UI
│       └── lime_explanation.png   # LIME explainability visualization
│
├── Ouputs/                        # Generated model outputs & embeddings
│   ├── cleaned_papers.parquet     # Cleaned paper dataset
│   ├── complete_dataset.parquet   # Full processed dataset
│   ├── metadata.json              # Dataset metadata
│   ├── vocabulary.json            # Extracted vocabulary
│   ├── sbert_abstract_embeddings.npy  # Sentence-BERT abstract embeddings
│   ├── sbert_title_embeddings.npy     # Sentence-BERT title embeddings
│   ├── scibert_embeddings.npy         # SciBERT embeddings
│   └── word2vec_embeddings.npy        # Word2Vec embeddings
│
├── evaluation/                    # Evaluation scripts and results (placeholder)
├── src/                           # Source modules (placeholder)
│
├── Individual-Final-Project-Report/   # Individual project reports (PDF)
├── Final Group Presentation/          # Final presentation slides (PDF)
├── Final Group Project Report/        # Final group report (PDF)
│
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## How It Works

### 1. Data Ingestion & Preprocessing
- **Input**: Papers from arXiv (cs.CL, cs.LG, stat.ML), ACL Anthology, and S2ORC
- **Process**: Download abstracts, titles, authors, and metadata → clean text → normalize → tokenize
- **Output**: `cleaned_papers.parquet`, `complete_dataset.parquet`

### 2. Embedding Generation
Multiple embedding models capture semantic meaning:
- **Word2Vec**: Traditional word embeddings for baseline
- **Sentence-BERT (SBERT)**: Dense embeddings for titles and abstracts
- **SciBERT**: Domain-specific embeddings for scientific text

### 3. Topic Modeling & Clustering
- Uses **BERTopic** with UMAP dimensionality reduction and HDBSCAN clustering
- Groups papers into coherent research topics
- Generates interpretable topic labels

### 4. Information Retrieval
- **BM25**: Keyword-based retrieval for exact matching
- **FAISS**: Vector similarity search using embeddings
- Hybrid approach combines both for optimal results

### 5. Classification & Summarization
- Zero-shot classification for adaptive categorization
- Transformer-based summarization for paper digests
- Named Entity Recognition (NER) for extracting key terms

### 6. Explainability & UI
- **LIME explanations**: Transparent feature importance for recommendations
- **Interactive UI** (`app.py`): Web interface for exploring papers and topics

---

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/adikan2k/Final-Project-Group-LexiCore.git
cd Final-Project-Group-LexiCore

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Pipeline

1. **Data Preprocessing**: Run `code/Week1_implementation_AdityaKanbargi.ipynb`
2. **Topic Modeling & Retrieval**: Run `code/Retrieval_Topic_Modeling.ipynb`
3. **Classification & Explainability**: Run `code/Week_3_Implementation_Trisha.ipynb`
4. **Launch UI**: 
   ```bash
   cd data/Trisha_Week_3
   python app.py
   ```

---

## Methods & Tools

| Component | Tools |
|-----------|-------|
| **Embeddings** | Word2Vec, Sentence-BERT, SciBERT |
| **Topic Modeling** | BERTopic, UMAP, HDBSCAN |
| **Retrieval** | BM25, FAISS |
| **Classification** | Zero-shot (Transformers) |
| **Explainability** | LIME |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## Evaluation Metrics

- **Classification**: Precision, Recall, F1-score
- **Topic Modeling**: Topic coherence, perplexity
- **System-Level**: Coverage, latency
- **Qualitative**: Faculty surveys for usefulness and transparency

---

## Datasets

| Dataset | Description |
|---------|-------------|
| **arXiv** | CS papers (cs.CL, cs.LG, stat.ML) — abstracts, categories, authors |
| **ACL Anthology** | Structured NLP venue metadata and citations |
| **S2ORC** | Full-text papers with citation graphs |

**Links**:
- [arXiv Bulk Data](https://arxiv.org/help/bulk_data)
- [ACL Anthology](https://aclanthology.org/)
- [S2ORC](https://allenai.org/data/s2orc)

---

## Team

- **Aditya Kanbargi**
- **Trisha Singh**
- **Pramod Krishnachari**

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.



