# Fake News Detector

A Machine Learning and NLP-based Fake News Detection web application built using Flask.

## Features

- Detects whether a news article is Real or Fake
- NLP preprocessing using NLTK
- TF-IDF Vectorization
- Logistic Regression model
- Interactive Flask web interface
- Real News shown in Green
- Fake News shown in Red

---

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Scikit-learn
- NLTK
- Pandas
- NumPy

---

## Project Structure

```bash
fake_news_detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── notebook/
│   └── eda.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <your-github-repo-link>
```

### Navigate to Project Folder

```bash
cd fake_news_detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

---

## Run Flask App

```bash
python app.py
```

---

## Open in Browser

```bash
https://fake-news-detector-1-pcd1.onrender.com
```

---

## Model Information

- TF-IDF Vectorizer
- Logistic Regression Classifier
- NLP Text Cleaning
- Stopword Removal
- Stemming using PorterStemmer

---
