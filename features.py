"""
features.py
-----------
Script to transform raw expense data into TF-IDF features and labels.
- Configurable input dataset via --data
- Optional --no-save for temporary feature extraction
- Saves features, labels, and vectorizer for training/inference
- Tracks execution time
"""

import argparse
import os
import time
from typing import Tuple, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load expense data from CSV.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} is empty.")
    if "description" not in df.columns or "category" not in df.columns:
        raise ValueError(f"{file_path} must have 'description' and 'category' columns.")
    return df


def extract_features(df: pd.DataFrame) -> Tuple:
    """
    Convert descriptions to TF-IDF features and extract labels.

    Args:
        df (pd.DataFrame): DataFrame with columns 'description' and 'category'.

    Returns:
        Tuple: (X_features, y_labels, vectorizer)
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    X_features = vectorizer.fit_transform(df["description"])
    y_labels = df["category"]
    return X_features, y_labels, vectorizer


def save_features(X, y, vectorizer, save: bool = True) -> None:
    """
    Save features, labels, and vectorizer as PKLs.

    Args:
        X: Feature matrix.
        y: Labels.
        vectorizer: TF-IDF vectorizer.
        save (bool): If False, do not save files.
    """
    if not save:
        print("⚡ --no-save flag set, skipping PKL saving.")
        return

    os.makedirs("models", exist_ok=True)
    joblib.dump(X, "models/X_features.pkl")
    joblib.dump(y, "models/y_labels.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("💾 Saved: X_features.pkl, y_labels.pkl, tfidf_vectorizer.pkl in models/")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Generate TF-IDF features from expense CSV.")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--no-save", action="store_true", help="Do not save PKL files")
    args = parser.parse_args()

    print(f"📂 Loading data from: {args.data}")
    df = load_data(args.data)

    print("✨ Extracting TF-IDF features...")
    X, y, vectorizer = extract_features(df)

    save_features(X, y, vectorizer, save=not args.no_save)

    elapsed = time.time() - start_time
    print(f"✅ Feature extraction complete! Execution time: {elapsed:.2f} seconds")
    print(f"📦 Features shape: {X.shape}, Labels count: {len(y)}")


if __name__ == "__main__":
    main()

