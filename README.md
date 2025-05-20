# Multi-Class-Sentiment-Classification-of-Mental-Health-Statements

This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify text statements into seven mental health statuses based on social media posts (e.g., Reddit, Twitter).

**Goal**: Develop a reliable sentiment classification model to assist in early detection and support for mental health conditions.

## Objectives

- Build a classifier to detect mental health status from user statements.
- Compare multiple ML algorithms.
- Identify the best-performing approach.
- Support future chatbot or monitoring system integration.

## Classes Covered

The model classifies text into the following categories:

- Normal  
- Depression  
- Suicidal  
- Anxiety  
- Stress  
- Bipolar  
- Personality Disorder  

## 🧪 Machine Learning Models

The following models were trained and evaluated:

- Multinomial Naive Bayes (MNB)  
- Support Vector Machine (SVM - RBF Kernel)  
- Logistic Regression  
- Random Forest (Best Performer: 90% test accuracy)

## Methodology

1. **Preprocessing**:
   - Text cleaning (remove URLs, punctuation, normalize spacing)
   - Contraction expansion
   - Tokenization & Lemmatization
   - Stopword filtering (custom NLTK list)

2. **Feature Extraction**:
   - TF-IDF Vectorization with 5,000 features

3. **Data Balancing**:
   - Handled using `RandomOverSampler` from imbalanced-learn

4. **Model Evaluation**:
   - 60/20/20 Train/Validation/Test split
   - Accuracy and confusion matrix analysis

## Results

- **Random Forest** achieved **90% accuracy** on the test set.
- Confusion between similar classes like *Depression* and *Suicidal* indicates real-world overlap.
- Simpler models like Naive Bayes offer speed but limited nuance.

## Dataset

**Source**: [Kaggle – Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
