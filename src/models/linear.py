import pandas as pd 
import joblib
import os
from .. import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from ..preprocessor import create_preprocessor

def main(RESULTS_OUTPUT):
    """
    Main function to create the linear model. 
    
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
    
    # load data
    X_train = pd.read_csv(config.X_TRAIN_DATA)
    y_train = pd.read_csv(config.Y_TRAIN_DATA)
    
    # train model
    preprocessor = create_preprocessor()
    pipe_ridge = make_pipeline(preprocessor, Ridge())

    cv_ridge = pd.DataFrame(cross_validate(pipe_ridge, 
                                        X_train, 
                                        y_train, 
                                        scoring='r2', \
                                        return_train_score=True, cv=10))

    # save results
    cv_results = {'ridge': cv_ridge.agg(['mean', 'std']).round(3).T}
    print(cv_results)
    output_file = os.path.join(RESULTS_OUTPUT, 'cv_results_linear.joblib') 
    joblib.dump(cv_results, output_file)

if __name__ == "__main__":
    RESULTS_OUTPUT = config.RESULTS_OUTPUT_DIR
    os.makedirs(RESULTS_OUTPUT, exist_ok=True)
    main(RESULTS_OUTPUT)
