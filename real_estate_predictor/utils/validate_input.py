import pandas as pd


def transform_input(data):
    """
    Assumes data is of type dict containing a single observation
    """
    # Transform the input data into the format required by the model
    
    input_df = pd.DataFrame.from_dict(data, orient='index').T
    return data