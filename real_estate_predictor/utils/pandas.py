import pandas as pd

def pandas_read_filepath(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the file.
            Accepts csv, json, or parquet file formats.
            If the file is not in one of these formats, a ValueError will be raised.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the file.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a csv, json, or parquet file.")
    return df