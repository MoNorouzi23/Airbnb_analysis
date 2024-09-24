import pandas as pd 
import joblib
import os
from src import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from src.preprocessor import create_preprocessor

def main():
    """
    Main function to create the baseline model. 
    """
    # Check necessary data exists
    if not os.path.isfile(config.X_TRAIN_DATA):
        print(f"Error: The X_train data does not exist.")
        return  # exit function
    
    if not os.path.isfile(config.Y_TRAIN_DATA):
        print(f"Error: The y_train data does not exist.")
        return  # exit function
    
    # load data
    X_train = pd.read_csv(config.X_TRAIN_DATA)
    y_train = pd.read_csv(config.Y_TRAIN_DATA)
    
    # train model
    preprocessor = create_preprocessor()
    pipe_dummy = make_pipeline(preprocessor, DummyRegressor())
    cv_dummy = pd.DataFrame(cross_validate(pipe_dummy, X_train, y_train, return_train_score=True, scoring='r2'))

    cv_results = {'dummy': cv_dummy.agg(['mean', 'std']).round(3).T}    
    pipe_dummy.fit(X_train, y_train)
    
    # save results
    joblib.dump(pipe_dummy, config.DUMMY_PATH) 
    joblib.dump(cv_results, config.CV_DUMMY_PATH)

if __name__ == "__main__":
    os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    main()
