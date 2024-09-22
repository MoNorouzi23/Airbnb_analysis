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

    # Target Distributions 
    target_dist_plots = target_distribution(df)

    # Target Distributions (grouped)
    target_cat, target_num = target_dist_grouped(df)

    # Correlation Plot
    pear_corr_plot = corr_plot(df, exclude_cols=['id', 'host_id'], title='Pearson Correlations')
    spear_corr_plot = corr_plot(df, corr_type='spearman', exclude_cols=['id', 'host_id'], title='Spearman Correlations')
    combined_corr = alt.hconcat(pear_corr_plot, spear_corr_plot)
    
    # Save plots
    cat_barplots.save(os.path.join(OUTPUT_PATH, "categorical_barcharts.png"))
    num_density_plots.save(os.path.join(OUTPUT_PATH, "numerical_density_plots.png"))
    target_dist_plots.save(os.path.join(OUTPUT_PATH, "target_dist_plots.png"))
    target_cat.save(os.path.join(OUTPUT_PATH, "target_dist_grouped_cat.png"))
    target_num.save(os.path.join(OUTPUT_PATH, "target_dist_grouped_num.png"))
    combined_corr.save(os.path.join(OUTPUT_PATH, 'correlation_plot.png'))

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

def target_distribution(df): 
    """
    Creates a histrogram and box plot for the target variable in the dataframe (df). 

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the target variable 'price'.
       
    Returns
    -------
    alt.Chart 
        The combined plots for the distribution of the target variable. 
    """
    df_filtered = df[df['price'] <= 4000]

    histogram = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('price:Q', bin=alt.Bin(step=70), 
                scale=alt.Scale(domain=(1, 4000)), 
                title='Price ($/night)'),
        y='count()', 
    ).properties(
        title='Distribution of Listing Prices', 
        width=400, 
        height=250
    )

    price_box_plot = alt.Chart(df).mark_boxplot().encode(
    x=alt.X('price:Q', title='Price ($/night)')
    ).properties(
        title="Box Plot for Listing Prices", 
        width=400,
        height=250
    )
    combined_plot = alt.hconcat(histogram, price_box_plot)
    return combined_plot

def target_dist_grouped(df):
    """
    Creates distribution plots for the target variable in the dataframe (df) grouped by key features in the dataframe. 

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the target variable 'price'.
       
    Returns
    -------
    tuple
        The combined plots for the distribution of the target variable as a tuple. 
    """
    # Plot grouped by Room Type and Neighboorhood group
    roomtype_price = alt.Chart(df, title='Price Distribution Based on Room Type').mark_boxplot().encode(
    x=alt.X('price:Q', title='Price'), 
    y=alt.Y('room_type:N', axis=alt.Axis(title='Room Type')) 
    ).properties(
        height=200,
        width=400
    )
    neighborhood_price = alt.Chart(df, title = 'Price Distribution Based on Neighborhood').mark_boxplot().encode(
        x = alt.X('price:Q', title = 'Price'), 
        y = alt.Y('neighbourhood_group').axis(title = 'Neighbourhood Group') 
    ).properties(
        height = 200,
        width = 400 
    )
    reviews_price = alt.Chart(df, title = 'Price Distribution Based on Number of Reviews').mark_bar().encode(
        x = alt.X('price').axis(title = 'Price'), 
        y = alt.Y('number_of_reviews').axis(title = 'Number of Reviews') 
    ).properties(
        height = 200,
        width = 400 
    )
    combined_cat = (roomtype_price | neighborhood_price)

    # Plot grouped by Number of Reviews
    reviews_price_scatter = alt.Chart(df, title='Price Distribution Based on Number of Reviews').mark_circle(opacity=0.6, size=60).encode(
    y=alt.Y('number_of_reviews:Q', title='Number of Reviews'),
    x=alt.X('price:Q', title='Price'),
    color=alt.Color('number_of_reviews:Q', scale=alt.Scale(scheme='viridis'), title='Number of Reviews'),
    tooltip=['number_of_reviews:Q', 'price:Q']
    ).properties(
        height=200,
        width=400
    ).interactive()

    # Marginal histogram for price
    hist_price = alt.Chart(df).mark_bar().encode(
        x=alt.X('price:Q', bin=alt.Bin(maxbins=30), title='Price'),
        y=alt.Y('count():Q', title='Count')
    ).properties(
        height=100,
        width=400
    )

    # Marginal histogram for number of reviews
    hist_reviews = alt.Chart(df).mark_bar().encode(
        y=alt.Y('number_of_reviews:Q', bin=alt.Bin(maxbins=30), title='Number of Reviews'),
        x=alt.X('count():Q', title='Count')
    ).properties(
        height=200,
        width=100
    )

    # Combine Plots 
    combined_num = (( reviews_price_scatter | hist_reviews ) & hist_price)
    return (combined_cat, combined_num)

def corr_plot(df, corr_type='pearson', exclude_cols=None, title='Correlation Plot'):
    """
    Creates Pearson and Spearman correlation heatmaps for the numeric columns in the dataframe (df). 
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to plot.
    corr_type : str
        The type of correlation to perform (default is 'pearson'). 
    title : str
        Title of the plot (default is 'Correlation Plot')
    exclude_cols : list
        A list of the mumeric column names not to include in the correlation plots. 
       
    Returns
    -------
    alt.Chart
        The correlation heatmap.  
    """
    df_numeric = df.select_dtypes(include='number').drop(columns=exclude_cols)
    df_corr = df_numeric.corr(method=corr_type).stack().reset_index(name='corr')

    # heatmap plot
    corr_heatmap = alt.Chart(df_corr).mark_rect().encode(
        x=alt.X('level_0:O', title='Variable', axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelAngle=45)),
        y=alt.Y('level_1:O', title='Variable', axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('corr', title='Correlation').scale(domain=(-1, 1), scheme='redblue', reverse=True)
    ).properties(
        width=250,
        height=250, 
        title=alt.TitleParams(text=title, fontSize=20)
    )
    # add correlation values 
    text = alt.Chart(df_corr).mark_text(baseline='middle').encode(
        x=alt.X('level_0:O'),
        y=alt.Y('level_1:O'),
        text=alt.Text('corr:Q', format='.2f'),
    )
    corr_plot = corr_heatmap + text
    return corr_plot

if __name__ == "__main__":
    DATA_PATH = "data/AB_NYC_2019.csv" 
    OUTPUT_PATH = "output/img"  
    main(DATA_PATH, OUTPUT_PATH)
