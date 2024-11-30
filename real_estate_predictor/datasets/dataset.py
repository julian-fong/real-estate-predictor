import pandas as pd
from real_estate_predictor.config import LISTING_PARAMETERS, NEIGHBOURHOOD_PARAMETERS, NEIGHBOURHOOD_KEYS, NEIGHBOURHOODS

class Dataset():
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        
    def get_listings_dataset(format, verbose = True):
        pass
    
    def save_dataset(format):
        pass