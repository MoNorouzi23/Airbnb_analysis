import pandas as pd 
import joblib
import os
from src import config
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
   
def mae_convert_calc(model, X_test, y_test):
    """
    This function coverts the target values into their original units and calculates the MAE. 
    
    Parameters
    ----------   
    model : sklearn.Pipeline 
        The model to evaluate. 
    X_test : pd.DataFrame 
        The test set features.  
    y_test : pd.DataFrame 
        The test set target values.  

    Returns
    -------   
    mae : int
        The mean absolute error of the model in the original units.     
    """
    y_pred = model.predict(X_test)
    y_pred_convert = np.exp(y_pred)
    y_test_convert = np.exp(y_test)
    mae = mean_absolute_error(y_test_convert, y_pred_convert)
    print(f'MAE test score: {mae:.3f}') 
    return mae 

def main():
    """
    Main function to ochestrate the R^2 and MAE test score calculations.     
    """
    # Check files exist 
    if not os.path.isfile(config.RFECV_PATH):
        print(f"Error: The RFECV model does not exist.")
        return  # exit function
    
    if not os.path.isfile(config.X_TEST_DATA):
        print(f"Error: The X_test data does not exist.")
        return  # exit function
    
    if not os.path.isfile(config.Y_TEST_DATA):
        print(f"Error: The y_test data does not exist.")
        return  # exit function
    
    # Load files
    X_test = pd.read_csv(config.X_TEST_DATA)
    y_test = pd.read_csv(config.Y_TEST_DATA)
    pipe_rfecv = joblib.load(config.RFECV_PATH)
    pipe_dummy = joblib.load(config.DUMMY_PATH)
    pipe_linear = joblib.load(config.LINEAR_PATH)

    # calculate r2 
    final_r2 = pipe_rfecv.score(X_test, y_test)
    print(f'R² test score: {final_r2:.3f}')

    # calculate mae and compare models 
    mae_rfecv = mae_convert_calc(pipe_rfecv, X_test, y_test) 
    mae_dummy = mae_convert_calc(pipe_dummy, X_test, y_test) 
    mae_linear = mae_convert_calc(pipe_linear, X_test, y_test) 
    
    mae_comparison = pd.DataFrame(data={'Model': ['RFECV', 'Dummy', 'Linear'],
                                        'MAE': [mae_rfecv, mae_dummy, mae_linear]
    }).sort_values('MAE')

    # Save results
    np.save(config.FINAL_R2_PATH, np.array(final_r2))
    mae_comparison.to_csv(config.MAE_PATH, index=False)

if __name__ == "__main__":
    RESULTS_OUTPUT = config.RESULTS_OUTPUT_DIR  
    os.makedirs(RESULTS_OUTPUT, exist_ok=True)
    main(RESULTS_OUTPUT)
