# Final-Project-GroupAlpha
Project group for the NLP class

Scholarly Topic Navigator: Explainable Research Digest Pipeline

This project examines the critical challenge of academic information overload faced by faculty laboratories
navigating the rapidly expanding landscape of NLP publications. Monthly publication volumes across venues
including ArXiv, ACL, EMNLP, and other conferences present a significant barrier for researchers attempting to
maintain current knowledge while fulfilling teaching and research obligations. To address this challenge, we
propose developing an automated system utilizing Natural Language Processing techniques, specifically
employing advanced models such as Word2Vec embeddings, transformer-based architectures
(BERT/RoBERTa), and zero-shot classification methods. The system will incorporate modern neural
approaches complemented by NLP preprocessing and normalization techniques, implemented through
established libraries including TensorFlow, Hugging Face Transformers, and spaCy for optimal text processing
and tokenization.
The primary NLP tasks encompass text preprocessing, document clustering, text classification, information
retrieval, and summarization, with entity recognition and zero-shot classification enabling continuous
recommendation improvements based on faculty feedback. Performance evaluation will employ comprehensive
metrics including Precision, Recall, and F1-score for classification tasks, topic coherence and perplexity for
topic modeling assessment, coverage and latency for system-level performance, and qualitative evaluation
through faculty surveys. This robust evaluation framework ensures the system effectively identifies relevant
papers while minimizing information overload, ultimately facilitating timely awareness of emerging research
opportunities and enabling transparent recommendation logic through actionable insights for the academic
community.
About Dataset:
Links: https://arxiv.org/help/bulk_data, https://aclanthology.org/, https://allenai.org/data/s2orc
This project utilizes three complementary datasets that together provide comprehensive coverage of the NLP
research landscape. The ArXiv Computer Science Collections (cs.CL, cs.LG, stat.ML) serves as the primary
source for real-time paper intake, providing abstracts, categories, author information, submission dates, and
categorization labels, with temporal metadata enabling analysis of publication trends and timely processing of
emerging research. The ACL Anthology compiles high-quality conference and journal papers from Association
for Computational Linguistics venues, offering structured metadata including venue information, citations, and
author affiliations, making it ideal for establishing ground truth in topic modeling and quality assessment. The
Semantic Scholar Open Research Corpus (S2ORC) provides full-text papers with rich citation graphs across
multiple disciplines, including paper identifiers, full-text content, citation context, and semantic relationships,
enabling sophisticated analysis including citation-based recommendation explanations and trend detection
through reference patterns, making it a comprehensive resource for developing academic research navigation
applications and conducting semantic analysis of scholarly literature.

