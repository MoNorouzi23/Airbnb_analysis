import pandas as pd 
import joblib
import os
from src import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from src.preprocessor import create_preprocessor

def main():
    """
    Main function to develop and train the linear model. 
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
    pipe_ridge = make_pipeline(preprocessor, Ridge())

    cv_ridge = pd.DataFrame(cross_validate(pipe_ridge, 
                                        X_train, 
                                        y_train, 
                                        scoring='r2',
                                        return_train_score=True, cv=10))

 
    cv_results = {'Ridge': cv_ridge.agg(['mean', 'std']).round(3).T}
    print(cv_results)
    
    pipe_ridge.fit(X_train, y_train)

    # save results
    joblib.dump(cv_results, config.CV_LINEAR_PATH)
    joblib.dump(pipe_ridge, config.LINEAR_PATH) 

if __name__ == "__main__":
    os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    main()
