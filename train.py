# Import Libraries
import pandas as pd
import numpy as np
import re
import string
import pickle

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load Dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add Labels
fake["label"] = 0
true["label"] = 1

# Keep Only Needed Columns
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

fake = fake[["content", "label"]]
true = true[["content", "label"]]

# Combine Dataset
df = pd.concat([fake, true], axis=0)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Shuffle Dataset
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

print(df["label"].value_counts())

# Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []

for i in range(len(df)):

    # remove special characters
    review = re.sub('[^a-zA-Z]', ' ', df['content'].iloc[i])

    # lowercase
    review = review.lower()

    # tokenization
    review = review.split()

    # stemming + stopwords removal
    review = [
        stemmer.stem(word)
        for word in review
        if word not in stop_words
    ]

    # join words
    review = " ".join(review)

    corpus.append(review)

# Features and Labels
X = corpus
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save Model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/vectorizer.pkl", "wb"))
print("Model and Vectorizer Saved Successfully")