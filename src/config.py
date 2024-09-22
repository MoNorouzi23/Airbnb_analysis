# Paths used in Scripts
import os 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(ROOT_DIR, 'data', 'AB_NYC_2019.csv')
FEAT_ENG_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'feature_engineered.csv')
X_TRAIN_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'X_train.csv')
Y_TRAIN_DATA = os.path.join(ROOT_DIR, 'data', 'output', 'y_train.csv')

IMG_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'img')
DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'output')
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'models')
RESULTS_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'results')
