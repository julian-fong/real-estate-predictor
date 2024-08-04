import pandas as pd

def extract_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts relevant data from the raw dataframe.
    Parameters
    ----------
    
    df : pd.DataFrame
        Incoming raw dataframe containing listings from a specific
        period. Assumes the following columns exists
        - details
        - address
        - condominium
        - nearby
        - map
    
    Returns
    -------
    df : pd.DataFrame
        Extracted data
    
    """
    
    keys = ['details', 'address', 'condominium', 'nearby','map']
    
    for key in keys:
        if key not in df.columns:
            raise ValueError(f"missing key {key} in dataframe columns")
        
    details_df = pd.DataFrame.from_records(df['details'])
    address_df = pd.DataFrame.from_records(df['address'])
    condo_df = pd.DataFrame.from_records(df['condominium'])
    nearby_df = pd.DataFrame.from_records(df['nearby'])
    map_df = pd.DataFrame.from_records(df['map'])

    df['city'] = address_df['city']
    df['area'] = address_df['area']
    df['district'] = address_df['district']
    df['neighborhood'] = address_df['neighborhood']
    df['zip'] = address_df['zip']

    df['latitude'] = map_df['latitude']
    df['longitude'] = map_df['longitude']

    df['fees'] = condo_df['fees']
    df['condo_ammenities'] = condo_df['ammenities']

    df['ammenities'] = nearby_df['ammenities']

    df['numBathrooms'] = details_df['numBathrooms']
    df['numBedrooms'] = details_df['numBedrooms']
    df['style'] = details_df['style']
    df['numKitchens'] = details_df['numKitchens']
    df['numRooms'] = details_df['numRooms']
    df['numParkingSpaces'] = details_df['numParkingSpaces']
    df['sqft'] = details_df['sqft']

    df['description'] = details_df['description']
    df['extras'] = details_df['extras']
    df['propertyType'] = details_df['propertyType']
    df['numGarageSpaces'] = details_df['numGarageSpaces']
    df['numDrivewaySpaces'] = details_df['numDrivewaySpaces']


    df = df.drop(columns=['details', 'address','condominium','map','nearby'])
    df = df.map(str)
    
    return df