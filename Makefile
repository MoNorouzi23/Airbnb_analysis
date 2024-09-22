# Makefile for running pipeline
.PHONY: all clean

# Default target
all: output/img/all_plots.png

# Command to generate all plots
output/img/all_plots.png: src/eda_plots.py data/AB_NYC_2019.csv
	python src/eda_plots.py --data_path=data/AB_NYC_2019.csv --output_path=output/img

# Clean up any generated files
clean:
	rm -rf output/img/*