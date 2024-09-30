import pandas as pd 
import numpy as np
import joblib
import os
from src import config
from src.preprocessor import create_preprocessor
import shap 
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


def main():
    """
    Main function to ochestrate generating the shap value plots. 
    """
    # Check files exist 
    files = [config.RFECV_PATH, config.X_TEST_DATA, config.Y_TEST_DATA, config.SELECTED_FEAT_PATH]
    for f in files: 
        if not os.path.isfile(f):
            print(f"Error no file: {f} ")
            return  # exit function
    
    # Load files
    X_test = pd.read_csv(config.X_TEST_DATA)
    y_test = pd.read_csv(config.Y_TEST_DATA)
    pipe_rfecv = joblib.load(config.RFECV_PATH)
    selected_features_mask, selected_features = joblib.load(config.SELECTED_FEAT_PATH)

    # Set up data
    preprocessor = pipe_rfecv.named_steps['columntransformer']
    data = preprocessor.transform(X_test)
    data_rfecv = data[:, selected_features_mask]
    X_test_enc = pd.DataFrame(data=data_rfecv, columns=selected_features, index=X_test.index)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(pipe_rfecv.named_steps["lgbmregressor"])
    shap_val = explainer.shap_values(X_test_enc)

    # Create & save plot 
    plt.figure()
    shap.summary_plot(shap_val, X_test_enc, show=False)
    plt.savefig(config.SHAP_SUM_PATH)
    plt.close()

    # Individual Prediction Plots  
    explanation = explainer(X_test_enc)

    # obtain examples of a low and high priced listing  
    y_test_reset = y_test.squeeze().reset_index(drop=True)
    avg_val = y_test_reset.mean()
    less_ind = y_test_reset[y_test_reset <= avg_val].index.tolist()
    gr_ind = y_test_reset[y_test_reset > avg_val].index.tolist()
    ex_less_ind = less_ind[100]
    ex_gr_ind = gr_ind[100]
    less_pred = pipe_rfecv.predict(X_test)[ex_less_ind] # a small predicted value 
    gr_pred = pipe_rfecv.predict(X_test)[ex_gr_ind] # a large predicted value 

    # Create & save plots
    index = [ex_less_ind, ex_gr_ind]
    path = [config.SHAP_LESS_PATH, config.SHAP_GR_PATH]
    for i, p in zip(index, path):
        plt.figure(figsize=(12, 8))  
        shap.plots.waterfall(explanation[i], show=False) 
        plt.savefig(p, bbox_inches='tight') 
        plt.close()

if __name__ == "__main__":
    os.makedirs(config.IMG_OUTPUT_DIR, exist_ok=True)
    main()
