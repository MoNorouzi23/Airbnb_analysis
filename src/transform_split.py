import pandas as pd
import numpy as np
from src import config
import os 
from sklearn.model_selection import train_test_split


def main(DATA_PATH, OUTPUT_PATH):
    """
    Main function to orchestrate target variable transformation and splitting the dataset into train and test sets. 
    
    Parameters
    ----------
    DATA_PATH : str
        The path to the feature engineered data.
    OUTPUT_PATH : str
        The path where to save the split datasets.
    """
    # Check feature engineered data exists
    if not os.path.isfile(DATA_PATH):
        print(f"Error: The feature engineered data does not exist.")
        return  # exit function

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    
    # log transform on target
    df['price'] = np.log(df['price'])

    # split data
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)
    X_train = train_df.drop(columns=['price'])
    y_train = train_df['price']
    X_test = test_df.drop(columns=['price'])
    y_test = test_df['price']

    # output data
    X_train.to_csv(os.path.join(OUTPUT_PATH, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_PATH, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_PATH, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(OUTPUT_PATH, 'y_test.csv'), index=False)

if __name__ == "__main__":
    DATA_PATH = config.FEAT_ENG_DATA
    OUTPUT_PATH = config.DATA_OUTPUT_DIR
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    main(DATA_PATH, OUTPUT_PATH)
