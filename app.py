import streamlit as st
import pandas as pd
import numpy as np
from src import config, preprocessing, features
import joblib
import lightgbm as lgb
import os

# Load resources (cache them for speed)
@st.cache_resource
def load_resources():
    tfidf = joblib.load(os.path.join(config.MODEL_DIR, 'tfidf_fast.pkl'))
    model_paths = [os.path.join(config.MODEL_DIR, f'lgb_fast_fold{i}.txt') for i in range(1, 6)]
    models = [lgb.Booster(model_file=p) for p in model_paths]
    return tfidf, models

st.title("🛒 Intelligent Product Price Predictor")
st.markdown("Enter a raw product description below to estimate its market value.")

# User Input
desc = st.text_area("Product Description", "Example: Nike Air Max running shoes, size 10, black/white, comfortable cushioning.")
image_link = st.text_input("Image Link (Optional)", "")

if st.button("Predict Price"):
    with st.spinner('Calculating...'):
        tfidf, models = load_resources()
        
        # Create a single-row dataframe
        df = pd.DataFrame({'catalog_content': [desc], 'image_link': [image_link]})
        
        # 1. Preprocess
        df = preprocessing.process_basic_features(df)
        
        # 2. Feature Engineering (Handle missing cols for single inference)
        # Note: For a real app, you'd load the brand mappings saved during training.
        # Here we use defaults for simplicity or you can save/load the brand_map dicts.
        df['brand_freq'] = 0
        df['brand_mean_oof'] = 23.0 # Global mean fallback
        df['token_mean_topk_avg_oof'] = 23.0 
        df['brand_ppu_oof'] = 0
        
        # 3. Vectorize
        X_text = tfidf.transform(df['catalog_clean'])
        
        # 4. Numeric
        num_cols = ['ipq','text_len','num_tokens','num_digits','has_image',
                    'brand_freq','brand_mean_oof','token_mean_topk_avg_oof','brand_ppu_oof']
        X_num = df[num_cols].fillna(-1).values
        
        from scipy.sparse import hstack, csr_matrix
        X_final = hstack([X_text, csr_matrix(X_num)])
        
        # 5. Predict (Average of folds)
        preds = [m.predict(X_final)[0] for m in models]
        avg_log_price = np.mean(preds)
        final_price = np.expm1(avg_log_price)
        
        st.success(f"Estimated Price: ${final_price:.2f}")