import pandas as pd
import numpy as np
import config
from datetime import datetime
import os 


def main(DATA_PATH, OUTPUT_PATH):
    """
    Main function to orchestrate clean the data and create the engineered features. 
    
    Parameters
    ----------
    DATA_PATH : str
        The path to the raw data.
    OUTPUT_PATH : str
        The path where to save the updated data.
    """
    # CLEANING
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Remove rows with price = 0 
    df = df[df['price'] != 0].copy()

    # Impute values 
    df.loc[:, 'reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df.loc[:, 'last_review'] = df['last_review'].fillna(pd.Timestamp('1900-01-01')) # to represent no previous reviews
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # FEATURE ENGINEERING
    df = estimated_listed_months(df)
    df = availability_ratio(df)
    df = days_since_last_review(df)
    df = distance_from_city_center(df)
    df.to_csv(os.path.join(OUTPUT_PATH, 'feature_engineered.csv'))


def estimated_listed_months(df): 
    """
    This function appends a new column 'estimated_listed_months' to the dataframe (df) 
    which provides a numeric value for the approximate time a listing has been listed.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe. 

    Returns
    -------
    pd.DataFrame
        The dataframe with the new column. 
    """
    df['estimated_listed_months'] = df['number_of_reviews'] / df['reviews_per_month']
    df['estimated_listed_months'].fillna(0, inplace=True)

    return df 

def availability_ratio(df): 
    """
    This function appends a new column 'availability_ratio' to the dataframe (df) 
    which provides a numeric value for proportion of the year a listing is available for booking.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe. 

    Returns
    -------
    pd.DataFrame
        The dataframe with the new column. 
    """
    df['availability_ratio'] = df['availability_365'] / 365

    return df 

def days_since_last_review(df): 
    """
    This function appends a new column 'days_since_last_review' to the dataframe (df) 
    which provides a numeric value for the number of days since the last review.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe. 

    Returns
    -------
    pd.DataFrame
        The dataframe with the new column. 
    """
    df['days_since_last_review'] = (datetime.now() - df['last_review']).dt.days
    df.loc[df['last_review'] == pd.Timestamp('1900-01-01'), 'days_since_last_review'] = 10000000000 # set listings with no reviews to a large number

    return df 


def distance_from_city_center(df): 
    """
    This function appends a new column distance_from_city_center' to the dataframe (df) 
    which provides a numeric value for the number of km  the listing is from the NYC center.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe. 

    Returns
    -------
    pd.DataFrame
        The dataframe with the new column. 
    """
    # calculate distance from city center (using haversine formula)
    nyc_center_lat = 40.7549
    nyc_center_lon = -73.9845

    def distance_from_city(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    df['distance_from_city_center'] = df.apply(lambda row: distance_from_city(
                row['latitude'], row['longitude'], nyc_center_lat, nyc_center_lon), axis=1)
        
    return df 

if __name__ == "__main__":
    DATA_PATH = config.RAW_DATA
    OUTPUT_PATH = config.DATA_OUTPUT_DIR
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    main(DATA_PATH, OUTPUT_PATH)
