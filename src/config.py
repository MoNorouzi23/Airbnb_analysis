# Paths used in Scripts
import os 

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(ROOT_DIR, 'data', 'AB_NYC_2019.csv')
IMG_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'img')
DATA_OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'output')
