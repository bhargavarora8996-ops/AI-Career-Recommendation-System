# 🎓 AI Student Career Intelligence & Recommendation System
### Two-Level Machine Learning & Career Recommendation Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.0.0-purple.svg)](https://plotly.com/)

---

## 📌 1. Project Overview & Two-Level Architecture

The **AI Student Career Intelligence System** is an upgraded, enterprise-grade Machine Learning and recommendation platform that analyzes student education level, age demographic, technical skills, and career interests to provide actionable career guidance.

The architecture operates via a **Two-Level System**:

- **LEVEL 1 (Supervised ML Classifier): Career Domain Prediction**
  Predicts one of 7 broader career domains (*Software & Systems*, *Data Science & AI*, *Data Engineering & Analytics*, *UI/UX & Design*, *Digital Marketing & Content*, *Finance & Management*, *Cloud & Cybersecurity*) using a leak-free Scikit-Learn `Pipeline([('preprocessor', ColumnTransformer(...)), ('classifier', LogisticRegression(balanced))])`.
  
- **LEVEL 2 (Career Recommendation Engine): Top-K Compatibility Ranking**
  Ranks fine-grained careers within the predicted domain (and dataset) using a multi-factor **Career Compatibility Score (0–100)** based on TF-IDF skill similarity, interest match, education compatibility, and ML domain confidence.

---

## 📊 2. Dataset Description & Diagnostic Analysis

Analysis of the Kaggle dataset (`AI-based Career Recommendation System.csv`):

- **Total Candidates (Rows):** 200
- **Total Attributes (Columns):** 8
- **Data Quality:** 0 missing values, 0 duplicate rows
- **Predictive Features ($X$):**
  - `Age` (MinMaxScaler)
  - `Education` (OneHotEncoder)
  - `Skills` (TfidfVectorizer on clean text tokens)
  - `Interests` (TfidfVectorizer on clean text tokens)
- **Excluded Non-Predictive / Leakage Columns:**
  - `CandidateID` & `Name` (Dropped)
  - `Recommendation_Score` (**Excluded from input feature matrix $X$ to prevent target leakage**)
- **Target Analysis & Class Support:**
  - **32 Fine-Grained Categories:** Average of ~6 candidates per class. 17 out of 32 classes have $\le 4$ samples.
  - **7 Career Domains:** Support ranges from 11 to 47 candidates per domain.

---

## 📈 3. ML Model Evaluation & Top-K Benchmarks

### A. Level 1: 7 Career Domains (Supervised ML Classifier)
Evaluated on 80/20 Stratified Train-Test Split with leak-free Scikit-Learn Pipelines:

| Model | Train Acc | Test Acc | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Overfitting Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression (balanced)** | **0.688** | **0.575** | **0.610** | **0.549** | **0.536** | **0.603** | **0.113** |
| **Multinomial Naive Bayes** | 0.706 | 0.600 | 0.547 | 0.503 | 0.510 | 0.580 | 0.106 |
| **Extra Trees (balanced)** | 0.844 | 0.525 | 0.555 | 0.500 | 0.503 | 0.545 | 0.319 |
| **Linear SVC (balanced)** | 0.713 | 0.525 | 0.525 | 0.500 | 0.490 | 0.556 | 0.188 |
| **Random Forest (balanced)** | 0.844 | 0.525 | 0.518 | 0.500 | 0.486 | 0.546 | 0.319 |

---

### B. Level 2 / Fine-Grained Classifier & Top-K Ranking Metrics (32 Classes)
Evaluated on 32 Fine-Grained Categories:

| Model | Test Accuracy (Top-1) | Top-3 Ranking Accuracy | Top-5 Ranking Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest (balanced)** | **42.5%** | **55.0%** | **60.0%** | **0.305** | **0.373** |
| **Logistic Regression (balanced)** | 40.0% | **60.0%** | **65.0%** | 0.301 | 0.354 |
| **Extra Trees (balanced)** | 37.5% | 57.5% | **65.0%** | 0.288 | 0.323 |

> **Top-K Rationale:** In career guidance, fine-grained titles share significant skill overlap (e.g. *Software Engineer* vs *Software Developer*). While single-class Top-1 accuracy is 42.5% (13.6x over 3.12% random guess), **Top-3 accuracy reaches 55.0%–60.0%**, confirming that the student's true path is successfully captured in top recommendations.

---

## ⚙️ 4. Transparent Career Compatibility Scoring Methodology

Level 2 Career Compatibility Scores are calculated using the following transparent weighted formula:

$$\text{Compatibility Score} = (0.45 \times \text{Skill Match \%}) + (0.30 \times \text{Interest Match \%}) + (0.15 \times \text{Education Match \%}) + (0.10 \times \text{Domain ML Fit \%})$$

- **Skill Match (45% Weight):** Proportion of candidate's skills matching dataset career profile skills.
- **Interest Match (30% Weight):** Proportion of candidate's interests matching dataset career profile interests.
- **Education Match (15% Weight):** Degree level compatibility based on dataset candidate distributions.
- **Domain ML Fit (10% Weight):** Level 1 ML classifier domain confidence percentage.

---

## 🎯 5. Streamlit Application & Features

- 🏠 **Home & Overview:** Key metrics, architecture explanation, and reliability warning banner.
- 🎯 **Career Predictor & Recommendations:** Level 1 Domain ML outcome + Level 2 Ranked Top 3 Career Cards (🥇, 🥈, 🥉) with compatibility scores, matched skills, missing skills, and dynamic XAI bullets.
- ⚖️ **Top Career Comparison:** Side-by-side comparison table and group bar chart comparing Top 3 careers across all 4 match factors.
- 🎯 **Skill Gap Analysis:** Match score progress bar, present skills chips, missing skills chips, and sample sufficiency warnings.
- 🗺️ **Personalized Roadmap & Projects:** Rule-based 6-step learning pathway and project suggestions.
- 📊 **Dataset Insights & EDA:** Distribution charts for 32 careers, 7 domains, education split, age histogram, top skills, top interests.
- 📈 **Model Performance & Top-K Metrics:** Benchmark tables for Level 1 & 2, Top-1/Top-3/Top-5 accuracy metrics, and 7-Domain Confusion Matrix heatmap.
- 📖 **About & Reliability Warnings:** Viva Q&A guide and scoring formula documentation.

---

## 🚀 6. How to Setup and Run

### 1. Activate Environment & Install Dependencies
```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Model Benchmarks & Export Artifacts
```bash
python src/train_model.py
```

### 3. Run Integration Test Suite
```bash
python test_pipeline.py
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🎓 7. Why Accuracy is Around 42.5% – 57.5% (Viva Guide)

1. **High Class Granularity vs Small Sample Size:** 200 total samples spread over 32 classes (~6 samples per class). 17 classes have $\le 4$ samples.
2. **Boundary Overlap:** Overlapping fine-grained roles (e.g. *Software Engineer* vs *Software Developer*) cause top-1 accuracy penalties even for near-identical predictions.
3. **Random Baseline Context:** Theoretical random guess is **3.12%**. Top-1 accuracy of **42.5%** is **13.6x higher than random chance**, while **Top-3 accuracy reaches 55.0%–60.0%**.
4. **Domain Grouping Impact:** Grouping 32 titles into 7 domains increases sample support (11 to 47 samples/domain), boosting **Test Accuracy to 57.5%** and **Macro F1 to 0.536**.
