import pandas as pd 
import joblib
import os
from .. import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor

from ..preprocessor import create_preprocessor


def rfecv_model_development(X_train, y_train): 
    """
    This function trains the rfecv model and provides the cross-validation results. 
    
    Returns
    -------   
    tuple
        A tuple containing: 
        - the cross-validation results (dict) 
        - the trained rfecv model (scikit-learn Pipeline object)
    """
    # hyperparameters
    alpha = 0.001
    lr = 0.08
    max_depth = 12
    num_leaves = 58
    reg_l = 100

    # model
    preprocessor = create_preprocessor()
    rfecv = RFECV(Lasso(alpha=alpha))

    pipe_rfecv = make_pipeline(
        preprocessor, rfecv, LGBMRegressor(random_state=123, 
                                           verbosity = 0, 
                                           learning_rate=lr, 
                                           max_depth=max_depth, 
                                           num_leaves=num_leaves, 
                                           reg_lambda=reg_l)
    )
    
    # obtain cv results 
    cv_rfecv = pd.DataFrame(cross_validate(pipe_rfecv, 
                                            X_train, 
                                            y_train,
                                            cv = 10, 
                                            return_train_score = True))
    cv_results = {'RFECV' : cv_rfecv.agg(['mean', 'std']).round(3).T}
    
    # fit entire training set 
    pipe_rfecv.fit(X_train, y_train)

    return (cv_results, pipe_rfecv, preprocessor)

def rfecv_feature_importances(model_rfecv, preprocessor, X_train):
    """
    This function identifies the selected features and feature importances from the RFECV model.  
    
    Returns
    -------   
    pd.DataFrame
        A pandas dataframe containing the selected features and their importances. 
    """
    rfecv_fs = model_rfecv.named_steps["rfecv"]
    selected_features_mask = rfecv_fs.support_

    # features names after preprocessing
    X_train_transformed = preprocessor.transform(X_train)

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(input_features=columns))
        else:
            feature_names.extend(columns)

    # Determine RFECV selected features
    selected_features = [feature for feature, selected in zip(feature_names, selected_features_mask) if selected]

    # obtain feature importances 
    model_lgbm = model_rfecv.named_steps['lgbmregressor']
    feature_importances_rfecv = model_lgbm.feature_importances_

    feat_imp_df = pd.DataFrame({'Feature': selected_features, 
                                          'Importance': feature_importances_rfecv}
                            ).sort_values(by='Importance', ascending=False)
    return feat_imp_df

def main(RESULTS_OUTPUT, MODEL_OUTPUT):
    """
    Main function to ochestrate training of the RFECV model and identifying feature importances. 
    
    Parameters
    ----------   
    RESULTS_OUTPUT : str
        The path to the directory for saving the cross-validation results and feature importances. 
    
    MODEL_OUTPUT : str
        The path to the directory for saving the model. 
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

    # create and train RFECV model 
    cv_results, model_rfecv, preprocessor = rfecv_model_development(X_train, y_train)

    # identify important features 
    feat_imp_df = rfecv_feature_importances(model_rfecv, preprocessor, X_train)

    # Save model, cv results, feature importances 
    output_file = os.path.join(MODEL_OUTPUT, 'model_rfecv.joblib') 
    joblib.dump(model_rfecv, output_file)

    output_file = os.path.join(RESULTS_OUTPUT, 'cv_results_RFECV.joblib') 
    joblib.dump(cv_results, output_file)

    output_file = os.path.join(RESULTS_OUTPUT, 'feat_imp_rfecv.csv') 
    feat_imp_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    RESULTS_OUTPUT = config.RESULTS_OUTPUT_DIR
    MODEL_OUTPUT = config.MODEL_OUTPUT_DIR
    os.makedirs(RESULTS_OUTPUT, exist_ok=True)
    main(RESULTS_OUTPUT, MODEL_OUTPUT)
