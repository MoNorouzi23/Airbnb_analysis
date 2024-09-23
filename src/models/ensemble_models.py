import pandas as pd 
import joblib
import os
from .. import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ..preprocessor import create_preprocessor

def main(RESULTS_OUTPUT):
    """
    Main function to create the ensemble models. 
    
    Parameters
    ----------   
    RESULTS_OUTPUT : str
        The path to the directory for saving the cross-validation results. 
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

    #Train RF 
    pipe_rf = make_pipeline(
    preprocessor, RandomForestRegressor(random_state=123, max_depth=10))

    cv_rf = pd.DataFrame(cross_validate(pipe_rf, X_train, y_train, cv=10, return_train_score=True))

    # Train XGboost 
    pipe_xgb = make_pipeline(preprocessor, XGBRegressor(
                    random_state=123, verbosity=0, max_depth=3, gamma=3, learning_rate=0.3))
    cv_xgb = pd.DataFrame(cross_validate(pipe_xgb, X_train, y_train, cv = 10, return_train_score = True))

    # Train LGBM
    pipe_lgbm = make_pipeline(preprocessor, LGBMRegressor(random_state=123, verbosity = 0))
    cv_lgbm = pd.DataFrame(cross_validate(pipe_lgbm, X_train, y_train, cv = 10, return_train_score = True))

    # Save results
    model_names = ['randomforest', 'XGBoost', 'LGBMRegressor']
    model_results = [cv_rf, cv_xgb, cv_lgbm]
    
    for name, result in zip(model_names, model_results): 
        cv_results = {name : result.agg(['mean', 'std']).round(3).T}
        output_file = os.path.join(RESULTS_OUTPUT, f'cv_results_{name}.joblib') 
        joblib.dump(cv_results, output_file)

if __name__ == "__main__":
    RESULTS_OUTPUT = config.RESULTS_OUTPUT_DIR
    os.makedirs(RESULTS_OUTPUT, exist_ok=True)
    main(RESULTS_OUTPUT)
