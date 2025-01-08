import re
from typing import List, Tuple, Union

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_data(path: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Loads data from file and splits it into train and test sets (80-20 split).
    Each row except first (header) contains Sentence and Sentiment separated by comma.
    Labels should be converted to integers: 1 for "positive" and 0 for "negative".

    Args:
        path: Path to file containing sentiment data

    Returns:
        Tuple containing:
        - List of training sentences
        - List of training labels (0 or 1)
        - List of test sentences
        - List of test labels (0 or 1)
    """
    df = pd.read_csv(path)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # Use train_test_split with fixed random_state for reproducibility
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return (
        train_df["sentence"].tolist(),
        train_df["sentiment"].tolist(),
        test_df["sentence"].tolist(),
        test_df["sentiment"].tolist(),
    )


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some tweet.

    Returns:
        String comprising the corresponding preprocessed tweet.
    """

    # Remove HTML
    bs = BeautifulSoup(doc, "html.parser")
    doc = " " + bs.get_text() + " "

    # Keep only letters
    doc = re.sub(r"[^a-zA-Z\s]", " ", doc)

    # Convert to lowercase
    doc = doc.lower()

    return doc


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return [preprocess(doc) for doc in docs]


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            tweet content.
        test_dataset: List of strings, each consisting of the preprocessed
            tweet content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for
          training and testing dataset respectively.
    """
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_dataset)
    X_test = vectorizer.transform(test_dataset)
    return X_train, X_test


def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model


def evaluate(y: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return recall, precision, f1, accuracy


if __name__ == "__main__":
    print("Loading data...")
    nltk.download("stopwords")
    train_data_raw, train_labels, test_data_raw, test_labels = load_data(
        "data/combined_sentiment_data.csv"
    )

    print("Processing data...")

    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
