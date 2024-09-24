# Makefile for running pipeline
.PHONY: all clean

# Default target
all: output/img/all_plots.png data/output/feature_engineered.csv data/output/data_split.csv output/models/models_refecv.joblib

# Generate EDA plots
output/img/all_plots.png: src/eda_plots.py src/config.py data/AB_NYC_2019.csv
	python -m src.eda_plots

# Clean data and create engineered features 
data/output/feature_engineered.csv: src/clean_and_engineer.py src/config.py data/AB_NYC_2019.csv
	python -m src.clean_and_engineer

data/output/data_split.csv: src/transform_split.py src/config.py data/output/feature_engineered.csv
	python -m src.transform_split

# Create RFECV Model 
output/models/models_refecv.joblib: src/models/rfecv.py src/config.py data/output/X_train.csv data/output/y_train.csv 
	python -m src.models.rfecv

# Clean up generated files
clean:
	rm -rf output/img/*
	rm -rf data/output/*
	rm -rf output/models/models_refecv.joblib
	rm -rf output/results/cv_results_RFECV.joblib
	rm -rf output/results/feat_imp_rfecv.csv 