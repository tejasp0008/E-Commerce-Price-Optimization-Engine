# src/config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

TRAIN_PATH = os.path.join(DATA_DIR, 'raw/train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'raw/test.csv')

# Hyperparameters
NUM_FOLDS = 5
SEED = 42
TF_WORD_MAX_FEATURES = 20000
TF_CHAR_MAX_FEATURES = 5000