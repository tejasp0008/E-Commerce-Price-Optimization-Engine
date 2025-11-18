# main.py
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src import config, preprocessing, features, training, utils

def main():
    # 1. Load Data
    print("Loading Data...")
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)

    # 2. Preprocessing
    train = preprocessing.process_text_features(train)
    test = preprocessing.process_text_features(test)

    # 3. Feature Engineering
    train, test = features.create_brand_features(train, test)
    
    # 4. Vectorization
    X_text_tr, X_text_te = features.create_tfidf_features(train['catalog_clean'], test['catalog_clean'])
    
    # 5. Combine Features (Simplify numeric selection for this example)
    num_cols = ['ipq', 'brand_freq'] # Add all your numeric cols
    X_num_tr = train[num_cols].fillna(-1).values
    X_num_te = test[num_cols].fillna(-1).values
    
    X_train = hstack([X_text_tr, csr_matrix(X_num_tr)])
    X_test = hstack([X_text_te, csr_matrix(X_num_te)])

    # 6. Train
    y_train = np.log1p(train['price']) # Log target
    oof, pred_test_log = training.train_lgbm(X_train, y_train, X_test)
    
    # 7. Evaluate
    oof_price = np.expm1(oof)
    score = utils.smape(train['price'], oof_price)
    print(f"Final CV SMAPE: {score}")

if __name__ == "__main__":
    main()