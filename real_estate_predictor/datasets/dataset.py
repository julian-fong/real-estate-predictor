import pandas as pd
from real_estate_predictor.utils.generate_dataset import (
    retrieve_repliers_listing_request, 
    retrieve_repliers_neighbourhood_request,
    save_dataset
    )
from real_estate_predictor.config.api_config import (
    LISTING_PARAMETERS, 
    NEIGHBOURHOOD_PARAMETERS, 
    NEIGHBOURHOOD_KEYS, 
    NEIGHBOURHOODS, 
    NEIGHBOURHOOD_NUMBEDROOMS, 
    NEIGHBOURHOOD_TYPES
    )

class Dataset():
    def __init__(self):
        pass
        
    def get_listings_dataset(self, start_date, end_date, verbose = False):
        self.listings_start_date = start_date
        self.listings_end_date = end_date
        
        _, numPages, _ = retrieve_repliers_listing_request(
            start_date, 
            end_date, 
            1, 
            LISTING_PARAMETERS, 
            verbose
        )
        
        listing_df = pd.DataFrame()
        
        for i in range(1, numPages+1):
            _, _, data = retrieve_repliers_listing_request(start_date, end_date, i, LISTING_PARAMETERS)
            df = pd.DataFrame(data['listings'])
            listing_df = pd.concat([listing_df, df], axis = 0)
            
        self.listings_df = listing_df
    
    def get_neighbourhood_dataset(self, start_date, end_date, verbose = False):
        self.neighbourhood_start_date = start_date
        self.neighbourhood_end_date = end_date
        
        
    
    def save_dataset(self, df, format, is_listings_dataset):
        save_dataset(df, format, is_listings_dataset)