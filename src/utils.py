# src/utils.py
import re
import numpy as np
import pandas as pd

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", s)
    s = re.sub(r'[^a-z0-9\.\,\-\s/x×]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    res = np.zeros_like(denom)
    res[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return np.mean(res) * 100.0