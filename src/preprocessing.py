# src/preprocessing.py
import re
import numpy as np
import pandas as pd
from .utils import clean_text

def extract_ipq(s):
    # [Paste your extract_ipq function logic here]
    pass 

def extract_unit_amount(s):
    # [Paste your extract_unit_amount function logic here]
    pass

def process_text_features(df):
    """Applies cleaning and regex extraction to the dataframe."""
    print("Processing text features...")
    df['catalog_clean'] = df['catalog_content'].fillna('').map(clean_text)
    df['ipq'] = df['catalog_clean'].map(extract_ipq)
    # Add other extractions...
    return df