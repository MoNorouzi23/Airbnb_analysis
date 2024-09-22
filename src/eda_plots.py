import pandas as pd
import altair as alt
import altair_ally as aly
import vegafusion
import os 

alt.data_transformers.enable("vegafusion")
aly.alt.data_transformers.enable('vegafusion')

def main(DATA_PATH, OUTPUT_PATH):
    """
    Main function to orchestrate generating the EDA plots and export plots to PNG files.
    
    Parameters
    ----------
    DATA_PATH : str
        The path to the dataset.
    OUTPUT_PATH : str
        The path where the plots will be saved.
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Categorical Columns Distributions
    categ_cols = ['room_type','neighbourhood_group']
    categ_names = ['Room Type', 'Neighbourhood Group']
    cat_barplots = cat_distributions(df, categ_cols, categ_names)

    # Numeric Column Distributions 
    exclude_cols = ['id', 'host_id', 'price']
    num_density_plots = num_distributions(df, exclude_cols)
    
    # Save plots
    cat_barplots.save(os.path.join(OUTPUT_PATH, "categorical_barcharts.png"))
    num_density_plots.save(os.path.join(OUTPUT_PATH, "numerical_density_plots.png"))

def cat_distributions(df, columns, titles):
    """
    Creates bar plots for given columns in the DataFrame (df).

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the columns to plot.
    columns : list
        A list of columns in df to plot distributions for.
    titles : list
        A list of names corresponding to the columns list, to display on plot title. 

    Returns
    -------
    alt.Chart
        Combined bar plot for the specified categorical columns 
    """
    categ_dist_plot = []
    for n, col in enumerate(columns):
        categ_cols_dist = alt.Chart(df, title = f'{titles[n]} Distribution').mark_bar().encode(
            y = alt.Y(col,type='nominal', title=titles[n]),
            x ='count()',
        ).properties(
            width = 400,
            height = 200
        )

        categ_dist_plot.append(categ_cols_dist)
    combined_plot = alt.hconcat(*categ_dist_plot)
    return combined_plot

def num_distributions(df, exclude_col=None):
    """
    Creates density plots for the numeric columns in the dataframe (df), except for columns specified, if any.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the columns to plot.
    exclude_col : list, optional
        A list of columns to not include in the plots.

    Returns
    -------
    alt.Chart
        The combined density plots for the numeric columns. 
    """
    df_numeric_feat = df.select_dtypes(include='number').drop(columns=exclude_col)
    num_dist_plot = aly.dist(df_numeric_feat).properties(
        title="Distributions of Numerical Features"
    )
    return num_dist_plot

if __name__ == "__main__":
    DATA_PATH = "data/AB_NYC_2019.csv" 
    OUTPUT_PATH = "output/img"  
    main(DATA_PATH, OUTPUT_PATH)
