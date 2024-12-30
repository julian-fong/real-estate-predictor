import pandas as pd
import time

from real_estate_predictor.utils.generate_dataset import (
    retrieve_repliers_listing_request, 
    retrieve_repliers_neighbourhood_request,
    save_dataset
    )
from real_estate_predictor.config.config import (
    LISTING_PARAMETERS, 
    NEIGHBOURHOOD_PARAMETERS, 
    NEIGHBOURHOOD_KEYS, 
    NEIGHBOURHOODS, 
    NEIGHBOURHOOD_NUMBEDROOMS, 
    NEIGHBOURHOOD_TYPES
    )

from real_estate_predictor.processing.extract_dataset import extract_neighbourhood_df, subtract_months

class Dataset():
    """
    Main class to construct the data pipeline for the datasets.
    
    Functionality:
        - Obtain raw listing and neighborhood datasets and save them into storage
        - Load listing and neighborhood datasets from storage or local paths
        - Merge the datasets into a single dataset for further processing
        - Select columns from the merged dataset to be used for training
        - Prepare the dataset using the Processor class
        - Create new features using the FeatureEngineering class
        - Save the final dataset into storage
        
        
    """
    def __init__(self):
        pass
        
    def get_listings_dataset(self, start_date, end_date, verbose = False):
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
        self.listings_df = listing_df
        
        return self.listings_df
    
    def get_neighbourhood_dataset(self, start_date, end_date, neighborhoods = None, verbose = False):
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
                    a = data['statistics']['soldPrice']['mth'] #gives avg, count and med
                    b = data['statistics']['listPrice']['mth'] #gives avg, count and med
                    c = data['statistics']['available']['mth'] #gives count 
                    d = data['statistics']['daysOnMarket']['mth'] #gives avg, count and med
                    e = data['statistics']['closed']['mth']

                    a_df = pd.DataFrame(a)
                    b_df = pd.DataFrame(b)
                    c_df = pd.DataFrame([c], index = ['count'])
                    d_df = pd.DataFrame(d)
                    e_df = pd.DataFrame(e)

                    for index in a_df.index:
                        a_df = a_df.rename(index = {index: f'{index}_{bed}_{type}_{location}'})
                        b_df = b_df.rename(index = {index: f'{index}_{bed}_{type}_{location}'})
                        d_df = d_df.rename(index = {index: f'{index}_{bed}_{type}_{location}_'})
                        
                    c_df = c_df.rename(index = {"count": f'count_{bed}_{type}_{location}'})

                    try:
                        a_df = extract_neighbourhood_df(a_df, "soldPrice")
                        b_df = extract_neighbourhood_df(b_df, "listPrice")
                        c_df = extract_neighbourhood_df(c_df, "available")
                        d_df = extract_neighbourhood_df(d_df, "daysOnMarket")

                        merged_inital_df = a_df.merge(b_df, on='key').merge(c_df, on='key').merge(d_df, on='key')

                        merged_inital_df['Date_1M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 1))
                        date_1m_df = merged_inital_df.merge(left_on = "Date_1M", right_on = "key", suffixes=[None, "L1M"], right = merged_inital_df, how = "left").drop(columns = ['keyL1M','Date_1ML1M','Date_1M'] + [col for col in merged_inital_df.columns if "current" in col])
                        merged_inital_df['Date_3M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 3))
                        date_3m_df = merged_inital_df.merge(left_on = "Date_3M", right_on = "key", suffixes=[None, "L3M"], right = merged_inital_df, how = "left").drop(columns = ['keyL3M','Date_1ML3M', 'Date_3ML3M','Date_1M','Date_3M'] + [col for col in merged_inital_df.columns if "current" in col])
                        merged_inital_df['Date_6M'] = merged_inital_df['key'].apply(lambda x: subtract_months(x, 6))
                        date_6m_df = merged_inital_df.merge(left_on = "Date_6M", right_on = "key", suffixes=[None, "L6M"], right = merged_inital_df, how = "left").drop(columns = ['keyL6M','Date_1ML6M', 'Date_3ML6M', 'Date_6ML6M','Date_1M','Date_3M','Date_6M'] + [col for col in merged_inital_df.columns if "current" in col])

                        merged_final_df = date_1m_df.merge(date_3m_df, on='key').merge(date_6m_df, on='key')
                        
                        df_stats = pd.concat([df_stats, merged_final_df], axis = 0)
                    
                    except:
                        if verbose:
                            print(f"unable to get data for {(location, bed, type)}")
                        continue
                        
                    if verbose:
                        print(f"Length of the stats dataset is: {len(df_stats)}")

        df_stats = df_stats.rename(columns = {"key": "neighborhood_key"})
        df_stats = df_stats.reset_index(drop = True)
        
        self.neighborhoods_df = df_stats
        return self.neighborhoods_df
        
    
    def save_raw_df(self, df, format, path = None, is_listings_dataset = True):
        save_dataset(df, format, path, is_listings_dataset)