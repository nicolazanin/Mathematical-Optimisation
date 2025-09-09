import pandas as pd
import numpy as np

def make_new_test_set(airports_number:int=5, seed:int=42)-> pd.DataFrame:
    """
    This function creates a dataframe with the airports test set.
    :param airports_number: number of airports we want to create in the test set file.
    :param seed: Seed for the random number generator.
    :return: test set dataframe.
    """

    np.random.seed(seed)
    coords_x = np.random.randint(0, 1000, size=airports_number)
    coords_y = np.random.randint(0, 1000, size=airports_number)

    airports_df = pd.DataFrame({
        'airport_id': range(airports_number),
        'x': coords_x,
        'y': coords_y
    })
    print_test_set(airports_df)
    make_new_test_set_file(airports_df)
    return airports_df

def make_new_test_set_file(airports_df:pd.DataFrame)-> None:
    """
    This function creates a new test set file with the airports dataframe.
    :param pd.DataFrame airports_df: pandas dataframe containing the airports test set.
    :return None:
    """
    file_name = 'airports.csv'
    airports_df.to_csv(file_name, index=False)
    print(f"File '{file_name}' successfully created with {len(airports_df)} airports.")


def print_test_set(airports_df:pd.DataFrame)-> None:
    """
    This function shows the airports test set.
    :param pd.DataFrame airports_df: pandas dataframe containing the airports test set.
    :return None:
    """

    print("\nAirports test set preview:")
    print(airports_df.head())

