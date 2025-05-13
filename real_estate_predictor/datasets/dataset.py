import pandas as pd
import time

from real_estate_predictor.utils.generate_dataset import (
    retrieve_repliers_listing_request, 
    retrieve_repliers_neighbourhood_request,
    save_dataset,
    save_raw_dataset,
    )
from real_estate_predictor.config.config import (
    LISTING_PARAMETERS, 
    NEIGHBOURHOOD_PARAMETERS, 
    NEIGHBOURHOOD_KEYS, 
    NEIGHBOURHOODS, 
    NEIGHBOURHOOD_NUMBEDROOMS, 
    NEIGHBOURHOOD_TYPES
    )

from real_estate_predictor.utils.extract_dataset import extract_neighborhood_data

class Dataset():
    """
    Main class to construct the data pipeline for the datasets.
    
    Available Functionality:
        - Get raw listing and neighborhood datasets and save them into storage
        - Load raw or transformed listing and neighborhood datasets from storage or local paths
        - Extract key columns from the raw datasets into a transformed state, and merge if necessary
        - Merge the datasets into a single dataset for further processing
        - Select columns from the merged dataset to be used for training
        - Prepare the dataset using the Processor class
        - Create new features using the FeatureEngineering class
        - Save any transformed/final dataset into storage
        
        
    """
    def __init__(self):
        pass
        
    def get_listings_dataset(self, start_date, end_date, save = False, verbose = False):
        self.listings_start_date = start_date
        self.listings_end_date = end_date
        
        _, numPages, _ = retrieve_repliers_listing_request(
            start_date = self.listings_start_date, 
            end_date = self.listings_end_date, 
            page_num = 1, 
            payload = LISTING_PARAMETERS,
            verbose = verbose
        )
        
        listing_df = pd.DataFrame()
        
        for i in range(1, numPages+1):
            _, _, data = retrieve_repliers_listing_request(start_date, end_date, i, LISTING_PARAMETERS, verbose=verbose)
            df = pd.DataFrame(data['listings'])
            listing_df = pd.concat([listing_df, df], axis = 0)
            
            #sleep for 5 seconds after every 5 pages to avoid rate limiting
            if i % 5 == 0:
                time.sleep(5)
                
            if verbose:
                print(f"Length of the listings dataset is: {len(listing_df)}")
        listing_df = listing_df.reset_index(drop = True)
        
        if save:
            self.save_raw_df(listing_df, "csv", path = None, is_listings_dataset = True)
        else:
            self.listings_df = listing_df
            return self.listings_df
    
    def get_neighbourhood_dataset(self, start_date, end_date, neighborhoods = None, save = False,verbose = False):
        self.neighbourhood_start_date = start_date
        self.neighbourhood_end_date = end_date
        self.neighborhoods = neighborhoods
        
        self._neighborhoods = NEIGHBOURHOODS if not neighborhoods else neighborhoods
        df_stats = pd.DataFrame()
        import time
        for bed in NEIGHBOURHOOD_NUMBEDROOMS:
            for type in NEIGHBOURHOOD_TYPES:
                for location in self._neighborhoods:
                    _, data = retrieve_repliers_neighbourhood_request(
                        self.neighbourhood_start_date,
                        self.neighbourhood_end_date,
                        NEIGHBOURHOOD_PARAMETERS,
                        type,
                        location,
                        bed,
                        verbose = verbose,
                    )
                    #sleep for 1 second after every request to avoid rate limiting
                    time.sleep(1)
                    
                    neighborhood_data = extract_neighborhood_data(data, location, bed, type, verbose)
                    
                    if isinstance(neighborhood_data, pd.DataFrame) and len(neighborhood_data) > 0:
                        df_stats = pd.concat([df_stats, neighborhood_data], axis = 0)    
                        
                    if verbose:
                        print(f"Length of the stats dataset is: {len(df_stats)}")

        df_stats = df_stats.rename(columns = {"key": "neighborhood_key"})
        df_stats = df_stats.reset_index(drop = True)
        if save:
            self.save_raw_df(df_stats, "csv", path = None, is_listings_dataset = False)
        else:
            self.neighborhoods_df = df_stats
            return self.neighborhoods_df
        
    
    def save_raw_df(self, df, format, path = None, is_listings_dataset = True):
        save_raw_dataset(df, format, path, is_listings_dataset)
        
    def save_df(self, df, format, path = None, file_name = None):
        save_dataset(df, format, path, file_name)
        
    def extract_and_merge_df(self, listing_df, neighborhood_df):
        pass
    
    def select_columns(self, df, columns):
        pass 
    
    def preprocess_dataset(self, df, columns):
        pass
    
    def feature_engineering(self, df, columns):
        pass
    
    def merge_datasets(self, listing_df, neighborhood_df):
        pass