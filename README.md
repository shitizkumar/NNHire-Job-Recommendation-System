# NNHire – Job Recommendation System

A hybrid job recommendation system that combines semantic NLP, skill matching, FAISS vector search, and machine learning to recommend relevant job opportunities based on a candidate's resume.

## Overview

NNHire analyzes a candidate's resume and compares it with available job postings using multiple signals such as:

* Semantic similarity between resume and job descriptions
* Job title similarity
* Required skill matching
* Preferred tool matching
* Experience compatibility
* Employment type
* Location compatibility

The system combines these signals to retrieve and rank relevant job opportunities.

## System Workflow

```text
Candidate Resume
       │
       ▼
Resume Text Extraction
       │
       ▼
Skill & Experience Extraction
       │
       ▼
Sentence Transformer Embeddings
       │
       ▼
Semantic Similarity
       │
       ├──────────────► Skill Matching
       │
       ├──────────────► Title Similarity
       │
       └──────────────► Job Attributes
                         │
                         ▼
                  FAISS Retrieval
                         │
                         ▼
                 Feature Engineering
                         │
                         ▼
                  XGBoost Ranker
                         │
                         ▼
                 Ranked Job Results
```

## Key Features

### 1. Resume Analysis

The system accepts a resume and extracts useful candidate information including:

* Skills
* Years of experience
* Job title
* Candidate profile information

### 2. Semantic Job Matching

Job descriptions and candidate information are converted into embeddings using:

`BAAI/bge-small-en-v1.5`

Cosine similarity is then used to measure semantic relevance between the candidate and job postings.

### 3. Skill Matching

The system compares candidate skills with:

* Required skills
* Preferred tools

This provides an additional matching signal beyond pure semantic similarity.

### 4. FAISS Vector Search

FAISS is used for efficient similarity search over job embeddings.

The implementation uses a normalized embedding index with `IndexFlatL2` to retrieve the most relevant job postings.

### 5. Machine Learning Ranking

The recommendation pipeline uses XGBoost with engineered features including:

* Semantic score
* Skill score
* Employment type score
* Experience score
* Location score

The model generates probability-based scores that are used in the final recommendation ranking.

## Model Evaluation

The classification/ranking pipeline was evaluated using both traditional classification metrics and recommendation-oriented metrics.

| Metric                         |     Result |
| ------------------------------ | ---------: |
| Accuracy                       |    **98%** |
| Precision                      |   **0.98** |
| Recall                         |   **0.98** |
| F1-score                       |   **0.98** |
| ROC-AUC                        | **0.9946** |
| Precision@20                   | **1.0000** |
| Recall@20                      | **0.0625** |
| Top-20 Recommendation Accuracy | **1.0000** |

The classification report was generated on a test set containing **640 samples**, with 0.98 accuracy and 0.98 macro/weighted F1-score.

> Note: Precision@20 of 1.0 means the evaluated top-20 set contained only relevant recommendations under the notebook's evaluation definition. Recall@20 was 0.0625, indicating that the top 20 captured 6.25% of the relevant items in the evaluated test set.

## Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### NLP & Embeddings

* Sentence Transformers
* BAAI/bge-small-en-v1.5
* Cosine Similarity

### Recommendation & Search

* FAISS

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Resume Processing

* PyMuPDF

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Google Colab
* Jupyter Notebook

## Dataset

The project uses an `ai_job_market.csv` dataset containing job-market information such as:

* Job ID
* Company
* Industry
* Job title
* Required skills
* Preferred tools
* Experience level
* Employment type
* Location
* Salary range
* Posted date
* Company size

The dataset is not included in this repository.

If you have permission to use the dataset, place it inside:

```text
data/
└── ai_job_market.csv
```

Do not upload private, sensitive, licensed, or non-redistributable datasets to GitHub.

## Project Structure

```text
NNHire-Job-Recommendation-System/
│
├── notebooks/
│   └── job_recommendation.ipynb
│
├── data/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How It Works

The recommendation process can be summarized as:

```text
Resume
   ↓
Extract candidate profile
   ↓
Generate resume embeddings
   ↓
Generate job embeddings
   ↓
Calculate semantic similarity
   ↓
Calculate skill/title similarity
   ↓
Retrieve candidates using FAISS
   ↓
Create ranking features
   ↓
XGBoost prediction
   ↓
Rank jobs
   ↓
Top recommended jobs
```

## Current Limitations

This repository currently contains the original Colab-based implementation.

Some parts of the notebook still depend on the Google Colab environment, including:

* Google Drive dataset paths
* Colab file upload
* Notebook execution order
* Interactive notebook state

The current notebook should therefore be treated as the original project implementation rather than a fully packaged production application.

## Future Improvements

* Convert the notebook into a modular Python application
* Add a Streamlit interface
* Add real-time resume upload
* Build a reusable recommendation pipeline
* Save and load the FAISS index
* Save trained ML models
* Improve recommendation evaluation
* Add automated tests
* Add configurable candidate preferences
* Deploy the recommendation API
* Add explainable recommendations showing why a job was recommended

## Project Goal

The goal of NNHire is to move beyond simple keyword-based job matching by combining semantic understanding with structured candidate-job compatibility signals.

The project demonstrates an end-to-end recommendation workflow using NLP embeddings, vector search, feature engineering, and machine learning ranking.

## Author

**Shitiz Kumar**

AI Engineer | Machine Learning | NLP | Generative AI

GitHub: [ShitizKumar](https://github.com/shitizkumar)
