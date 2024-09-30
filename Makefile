# Makefile for running pipeline
.PHONY: all clean

# Default target
all: output/img/all_plots.png data/output/feature_engineered.csv data/output/correlation_updated_plot.png data/output/data_split.csv output/models/models_rfecv.joblib output/img/shap_gr_plot.png 

# Generate EDA plots
output/img/all_plots.png: src/eda_plots.py src/config.py data/AB_NYC_2019.csv
	python -m src.eda_plots

# Clean data and create engineered features 
data/output/feature_engineered.csv: src/clean_and_engineer.py src/config.py data/AB_NYC_2019.csv
	python -m src.clean_and_engineer

# Obtain updated correlations
output/img/correlation_updated_plot.png: src/updated_correlations.py data/output/feature_engineered.csv src/config.py
	python -m src.updated_correlations

# Split training/test and feature/target	
data/output/data_split.csv: src/transform_split.py src/config.py data/output/feature_engineered.csv
	python -m src.transform_split

# Create RFECV model 
output/models/models_rfecv.joblib output/results/selected_feat.joblib output/results/feat_imp_rfecv.csv: src/models/rfecv.py src/config.py data/output/X_train.csv data/output/y_train.csv 
	python -m src.models.rfecv

# Evaluate RFECV model
output/results/final_r2.npy output/results/mae_comparison.csv: src/evaluate.py src/config.py data/output/X_test.csv \
	data/output/y_test.csv \
    output/models/model_rfecv.joblib \
	output/models/model_linear.joblib \
    output/models/model_dummy.joblib
	python -m src.evaluate

# Create SHAP plots
output/img/shap_gr_plot.png output/img/shap_less_plot.png output/img/shap_summary.png: src/shap_values.py src/config.py \
	data/output/X_test.csv \
	data/output/y_test.csv \
    output/models/model_rfecv.joblib 
	python -m src.shap_values 

# Clean up generated files
clean:
	rm -rf output/img/*
	rm -rf data/output/*
	rm -rf output/models/model_rfecv.joblib
	rm -rf output/results/cv_results_RFECV.joblib
	rm -rf output/results/selected_feat.joblib
	rm -rf output/results/feat_imp_rfecv.csv 
	rm -rf output/results/final_r2.npy
	rm -rf output/results/mae_comparison.csv

