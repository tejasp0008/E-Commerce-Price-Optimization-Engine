# src/features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from . import config

def create_brand_features(train_df, test_df):
    """Calculates brand frequency and OOF brand mean price."""
    # [Paste your Brand Frequency and OOF Mean logic here]
    return train_df, test_df

def create_tfidf_features(train_text, test_text):
    """Generates TF-IDF matrices."""
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=config.TF_WORD_MAX_FEATURES, ngram_range=(1,2), min_df=5)
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    
    # Save vectorizer for inference
    joblib.dump(tfidf, os.path.join(config.MODEL_DIR, 'tfidf.pkl'))
    
    return X_train, X_test