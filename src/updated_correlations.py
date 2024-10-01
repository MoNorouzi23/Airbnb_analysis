import pandas as pd
import altair as alt
import altair_ally as aly
import vegafusion
import os 
from src import config
from src.eda_plots import corr_plot

alt.data_transformers.enable("vegafusion")
aly.alt.data_transformers.enable('vegafusion')

def main():
    """
    Main function to plot the updated correlations. 
    """
    df = pd.read_csv(config.FEAT_ENG_DATA)
    
    # Correlation Plot
    pear_corr_plot = corr_plot(df, exclude_cols=['id', 'host_id', 'availability_365', 'number_of_reviews'], title='Pearson Correlations')
    spear_corr_plot = corr_plot(df, corr_type='spearman', exclude_cols=['id', 'host_id', 'availability_365', 'number_of_reviews'], title='Spearman Correlations')
    combined_corr = alt.hconcat(pear_corr_plot, spear_corr_plot)
    
    # Save plots
    combined_corr.save(config.CORR_UPDATED_PATH) 

if __name__ == "__main__":
    os.makedirs(config.IMG_OUTPUT_DIR, exist_ok=True)
    main()