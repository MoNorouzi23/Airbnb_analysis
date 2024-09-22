# Makefile for running pipeline
.PHONY: all clean

# Default target
all: output/img/all_plots.png data/output/feature_engineered.csv

# Generate EDA plots
output/img/all_plots.png: src/eda_plots.py src/config.py data/AB_NYC_2019.csv
	python src/eda_plots.py

# Generate clean data and engineered features 
data/output/feature_engineered.csv: src/clean_and_engineer.py src/config.py data/AB_NYC_2019.csv
	python src/clean_and_engineer.py 

# Clean up generated files
clean:
	rm -rf output/img/*
	rm -rf data/output/feature_engineered.csv