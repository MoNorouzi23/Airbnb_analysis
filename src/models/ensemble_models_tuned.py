import pandas as pd 
import numpy as np
import joblib
import os
from src import config
from src.preprocessor import create_preprocessor 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import uniform, randint


def main():
    """
    Main function to ochestrate the hyperparameter optimization for the ensemble models. 
    """
    # Check files exist 
    files = [config.X_TRAIN_DATA, config.Y_TRAIN_DATA]
    for f in files: 
        if not os.path.isfile(f):
            print(f"Error no file: {f} ")
            return  # exit function
    
    # Load files
    X_train = pd.read_csv(config.X_TRAIN_DATA)
    y_train = pd.read_csv(config.Y_TRAIN_DATA)
    y_train = y_train.values.ravel()

    preprocessor = create_preprocessor()

    # RF TUNING 
    pipe_rf = make_pipeline(preprocessor, RandomForestRegressor())

    param_dist = {
    'randomforestregressor__max_depth': [5,10,15],
    'randomforestregressor__n_estimators' : [100,200,500],
    'randomforestregressor__min_samples_leaf' : [1,2,4], 
    }

    random_search_rf = RandomizedSearchCV(estimator=pipe_rf, 
                                        param_distributions=param_dist,
                                        n_iter=50, 
                                        scoring='r2',
                                        cv=2, 
                                        n_jobs=-1, 
                                        random_state=123, 
                                        return_train_score=True)

    random_search_rf.fit(X_train, y_train)

    # Obtain best model results 
    cv_results_rf = pd.DataFrame(random_search_rf.cv_results_)
    best_index = random_search_rf.best_index_
    best_results = cv_results_rf.loc[best_index]
    results = best_results[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]
    results_rf = pd.DataFrame({
        'mean': [results['mean_test_score'], results['mean_train_score']],
        'std': [results['std_test_score'], results['std_train_score']]
    }, index=['test_score', 'train_score']).round(3)

    best_params_rf = random_search_rf.best_params_
    
    print("Best Hyperparameters:")
    for param, value in best_params_rf.items(): 
        param_name = param.replace('randomforestregressor__', '')
        print(f" - {param_name}: {value}")

    results_rf_dict = {'RF_Tuned': results_rf}
    joblib.dump(results_rf_dict, config.CV_RF_TUNED_PATH)

    # XGBOOST TUNED
    pipe_xgb = make_pipeline(preprocessor, XGBRegressor())

    param_dist = {
    'xgbregressor__learning_rate': uniform(loc=0.01, scale=0.3),
    'xgbregressor__n_estimators': randint(100, 500),
    'xgbregressor__max_depth': randint(3, 15),
    'xgbregressor__min_child_weight': randint(1, 10),
    'xgbregressor__alpha': uniform(loc=0, scale=10),  
    'xgbregressor__lambda': uniform(loc=0, scale=10)  

    }

    random_search_xgb = RandomizedSearchCV(estimator=pipe_xgb, 
                                    param_distributions=param_dist,
                                    n_iter=100, 
                                    scoring='r2', 
                                    cv=5, 
                                    n_jobs=-1, 
                                    random_state=123, 
                                    verbose=2, 
                                    return_train_score=True
                                    )

    random_search_xgb.fit(X_train, y_train)

    # Obtain best model results 
    cv_results_xgb = pd.DataFrame(random_search_xgb.cv_results_)
    best_index = random_search_xgb.best_index_
    best_results = cv_results_xgb.loc[best_index]
    results = best_results[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]
    results_xgb = pd.DataFrame({
        'mean': [results['mean_test_score'], results['mean_train_score']],
        'std': [results['std_test_score'], results['std_train_score']]
    }, index=['test_score', 'train_score']).round(3)


    best_params_xgb = random_search_xgb.best_params_
    print("Best Hyperparameters:")
    for param, value in best_params_xgb.items(): 
        param_name = param.replace('xgbregressor__', '')
        print(f" - {param_name}: {value}")


    results_xgb_tuned = {'XGB_Tuned': results_xgb}
    joblib.dump(results_xgb_tuned, config.CV_XGB_TUNED_PATH)

    # LGBM TUNED 
    pipe_lgbm = make_pipeline(preprocessor, LGBMRegressor())

    param_dist = {
    'lgbmregressor__learning_rate': uniform(loc=0.01, scale=0.3),  
    'lgbmregressor__num_leaves': randint(24, 80),  
    'lgbmregressor__max_depth': randint(3, 15),  
    'lgbmregressor__reg_lambda': [0, 1,10,100]
    }

    random_search_lgbm = RandomizedSearchCV(estimator=pipe_lgbm, 
                                    param_distributions=param_dist,
                                    n_iter=100, 
                                    scoring='r2', 
                                    cv=5, n_jobs=-1, 
                                    random_state=123, 
                                    return_train_score=True)

    random_search_lgbm.fit(X_train, y_train)

    # Obtain best model results 
    cv_results_lgbm = pd.DataFrame(random_search_lgbm.cv_results_)
    best_index = random_search_lgbm.best_index_
    best_results = cv_results_lgbm.loc[best_index]
    results = best_results[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]
    results_lgbm = pd.DataFrame({
        'mean': [results['mean_test_score'], results['mean_train_score']],
        'std': [results['std_test_score'], results['std_train_score']]
    }, index=['test_score', 'train_score']).round(3)

    results_lgbm_dict = {'LGBM_Tuned': results_lgbm}

    joblib.dump(results_lgbm_dict, config.CV_LGBM_TUNED_PATH)

    best_params_lgbm = random_search_lgbm.best_params_
    print("Best Hyperparameters")
    for param, value in best_params_lgbm.items(): 
        param_name = param.replace('lgbmregressor__', '')
        print(f"{param_name}: {value}")





if __name__ == "__main__":
    os.makedirs(config.IMG_OUTPUT_DIR, exist_ok=True)
    main()
