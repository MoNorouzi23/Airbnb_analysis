# Makefile for running pipeline
.PHONY: all clean

# Default target
all: output/img/categorical_barcharts.png output/img/numerical_density_plots.png

# Generate categorical distribution plots
output/img/categorical_barcharts.png: src/eda_plots.py data/AB_NYC_2019.csv
	python src/eda_plots.py --data_path=data/AB_NYC_2019.csv --output_path=output/img

# Generate numerical density plots
output/img/numerical_density_plots.png: src/eda_plots.py data/AB_NYC_2019.csv
	python src/eda_plots.py --data_path=data/AB_NYC_2019.csv --output_path=output/img

# clean generated files 
clean:
	rm -rf output/img/*
