import pandas as pd 
import joblib
import os
from src import config
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor

from src.preprocessor import create_preprocessor


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
    print(cv_results)
    
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
    return feat_imp_df, selected_features_mask, selected_features

def main():
    """
    Main function to ochestrate training of the RFECV model and identifying feature importances.    
    """
    # Check files exist 
    files = [config.X_TRAIN_DATA, config.Y_TRAIN_DATA]
    for f in files: 
        if not os.path.isfile(f):
            print(f"Error no file: {f} ")
            return  # exit function
    
    # Load data
    X_train = pd.read_csv(config.X_TRAIN_DATA)
    y_train = pd.read_csv(config.Y_TRAIN_DATA)
    y_train = y_train.values.ravel()

    # create and train RFECV model 
    cv_results, model_rfecv, preprocessor = rfecv_model_development(X_train, y_train)

    # identify important features 
    feat_imp_df, selected_features_mask, selected_features = rfecv_feature_importances(model_rfecv, preprocessor, X_train)

    # Save model, cv results, feature importances 
    joblib.dump(model_rfecv, config.RFECV_PATH)
    joblib.dump(cv_results, config.CV_RFECV_PATH)
    feat_imp_df.to_csv(config.FEAT_IMP_PATH, index=False)
    joblib.dump((selected_features_mask, selected_features), config.SELECTED_FEAT_PATH)

if __name__ == "__main__":
    os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    main()
