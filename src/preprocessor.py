from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

def create_preprocessor():
    """
    Creates the preprocessor for transforming features in the dataset.

    Returns
    -------
    ColumnTransformer
        A scikit-learn ColumnTransformer object that applies transformations 
        to the specified columns in the input dataframe.
    """
    categorical_features = ['room_type', 'neighbourhood_group']
    
    numeric_features = ['latitude', 'longitude', 'minimum_nights', 
        'calculated_host_listings_count', 'reviews_per_month', 
        'estimated_listed_months', 'availability_ratio', 
        'days_since_last_review', 'distance_from_city_center']
    
    drop_features = [ 'last_review', 'id', 'name', 'host_id', 
        'host_name', 'availability_365', 
        'number_of_reviews', 'neighbourhood']

    numeric_transformer = make_pipeline(StandardScaler())
    categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features), 
        ('drop', drop_features)
    )

    return preprocessor

def main():
    """
    Main function to create the preprocessor used in the models. 
    
    Parameters
    ----------
    None
    """
    preprocessor = create_preprocessor()

    # placeholder for future development 

if __name__ == "__main__":
    main()
