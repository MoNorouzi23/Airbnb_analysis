import pandas as pd 
import joblib
import os
from src import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.preprocessor import create_preprocessor

def main():
    """
    Main function to create the ensemble models. 
    """
    # Check necessary data exists
    if not os.path.isfile(config.X_TRAIN_DATA):
        print(f"Error: The X_train data does not exist.")
        return  # exit function
    
    if not os.path.isfile(config.Y_TRAIN_DATA):
        print(f"Error: The y_train data does not exist.")
        return  # exit function
    
    # Load data
    X_train = pd.read_csv(config.X_TRAIN_DATA)
    y_train = pd.read_csv(config.Y_TRAIN_DATA)
    y_train = y_train.values.ravel()
    
    preprocessor = create_preprocessor()   

    # Train RF 
    pipe_rf = make_pipeline(preprocessor, RandomForestRegressor(random_state=123, max_depth=10))
    cv_rf = pd.DataFrame(cross_validate(pipe_rf, X_train, y_train, cv=10, return_train_score=True))
    results_rf = {'Random Forest' : cv_rf.agg(['mean', 'std']).round(3).T}

    # Train XGboost 
    pipe_xgb = make_pipeline(preprocessor, XGBRegressor(random_state=123, verbosity=0, max_depth=3, gamma=3, learning_rate=0.3))
    cv_xgb = pd.DataFrame(cross_validate(pipe_xgb, X_train, y_train, cv = 10, return_train_score = True))
    results_xgb = {'XGBoost' : cv_xgb.agg(['mean', 'std']).round(3).T}

    # Train LGBM
    pipe_lgbm = make_pipeline(preprocessor, LGBMRegressor(random_state=123, verbosity = 0))
    cv_lgbm = pd.DataFrame(cross_validate(pipe_lgbm, X_train, y_train, cv = 10, return_train_score = True))
    results_lgbm = {'LGBM' : cv_lgbm.agg(['mean', 'std']).round(3).T}

    # Save results
    joblib.dump(results_rf, config.CV_RF_PATH)
    joblib.dump(results_xgb, config.CV_XGB_PATH)
    joblib.dump(results_lgbm, config.CV_LGBM_PATH)

if __name__ == "__main__":
    os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)
    main()
