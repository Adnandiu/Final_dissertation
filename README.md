README: Sentiment Analysis of Amazon Product Reviews

README: Sentiment Analysis of Amazon Product Reviews
A Comparative Study of Machine Learning and Deep Learning Approaches


This notebook, Final_edation.ipynb, is part of my MSc dissertation project:
"Sentiment Analysis of Amazon Product Reviews using Python: A Comparative Study of Machine Learning and Deep Learning Approaches."

The goal is to analyze customer reviews from Amazon and build both traditional ML and DL models to classify sentiments (positive/negative/neutral) — and compare their performance.

 What This Project Covers

- Data Preprocessing
  • Cleaning text data (removing stopwords, punctuation, etc.)
  • Tokenization and Lemmatization using NLTK
  • Word Cloud visualization for exploratory insights

- Sentiment Scoring
  • Sentiment polarity analysis using TextBlob

- Model Building
  • Machine Learning: TF-IDF + Multinomial Naive Bayes
  • Deep Learning: LSTM-based sentiment classifier with Keras

- Evaluation Metrics
  • Accuracy
  • Precision, Recall, F1-Score
  • Classification Reports for both ML and DL models

 Required Libraries

Install all required packages using:

pip install pandas numpy matplotlib seaborn nltk wordcloud textblob scikit-learn tensorflow

Download necessary NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

 How to Run

1. Make sure your environment has the required libraries installed.
2. Open and run the notebook step-by-step.
3. Load your dataset (Amazon reviews CSV with a text and label column).
4. Run preprocessing, sentiment analysis, and train both models.
5. Compare the performance of ML and DL approaches.

 Expected Outputs

- Word clouds of frequently used terms
- Sentiment polarity graphs
- Accuracy and evaluation reports of both models
- Model performance comparison

 Tools & Frameworks Used

- Python
- NLTK, TextBlob
- Scikit-learn (ML models)
- TensorFlow / Keras (DL models)
- Matplotlib, Seaborn

 Author

Yeasir Adnan
Id-30118447
Dissertation Title: Sentiment Analysis of Amazon Product Reviews using Python: A Comparative Study of Machine Learning and Deep Learning Approaches

