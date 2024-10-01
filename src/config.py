# Paths used in Scripts
import os 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 
# Directories
IMG_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'img')
DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'output')
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'models')
RESULTS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'results')

# Data 
RAW_DATA = os.path.join(ROOT_DIR, 'data', 'AB_NYC_2019.csv')
FEAT_ENG_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'feature_engineered.csv')
X_TRAIN_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'X_train.csv')
Y_TRAIN_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'y_train.csv')
X_TEST_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'X_test.csv')
Y_TEST_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'y_test.csv')

# Images
CAT_BAR_PATH = os.path.join(IMG_OUTPUT_DIR, "categorical_barcharts.png")
NUM_DENSITY_PATH = os.path.join(IMG_OUTPUT_DIR, "numerical_density_plots.png")
TAR_DIST_PATH = os.path.join(IMG_OUTPUT_DIR, "target_dist_plots.png")
TAR_CAT_PATH = os.path.join(IMG_OUTPUT_DIR, "target_dist_grouped_cat.png")
TAR_NUM_PATH =os.path.join(IMG_OUTPUT_DIR, "target_dist_grouped_num.png")
CORR_PATH = os.path.join(IMG_OUTPUT_DIR, 'correlation_plot.png')
CORR_UPDATED_PATH = os.path.join(IMG_OUTPUT_DIR, 'correlation_updated_plot.png')
SHAP_SUM_PATH = os.path.join(IMG_OUTPUT_DIR, 'shap_summary.png')
SHAP_LESS_PATH = os.path.join(IMG_OUTPUT_DIR, 'shap_less_plot.png')
SHAP_GR_PATH = os.path.join(IMG_OUTPUT_DIR, 'shap_gr_plot.png')

# Models 
RFECV_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model_rfecv.joblib')
DUMMY_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model_dummy.joblib')
LINEAR_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model_linear.joblib')

# Results
CV_RFECV_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_RFECV.joblib') 
CV_LINEAR_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_linear.joblib') 
CV_DUMMY_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_dummy.joblib') 
CV_XGB_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_xgb.joblib') 
CV_LGBM_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_lgbm.joblib') 
CV_RF_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_rf.joblib') 
CV_RF_TUNED_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_rf_tuned.joblib') 
CV_XGB_TUNED_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_xgb_tuned.joblib') 
CV_LGBM_TUNED_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'cv_results_lgbm_tuned.joblib') 
FEAT_IMP_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'feat_imp_rfecv.csv') 
SELECTED_FEAT_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'selected_feat.joblib') 
FINAL_R2_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'final_r2.npy') 
MAE_PATH = os.path.join(RESULTS_OUTPUT_DIR, 'mae_comparison.csv')