# src/training.py
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .utils import smape
from . import config
import os

def train_lgbm(X, y, X_test):
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(X_test.shape[0])
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 64,
        # ... other params
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training Fold {fold + 1}...")
        # [Paste your training loop logic here]
        # Save models to config.MODEL_DIR
        
    return oof_preds, test_preds