from real_estate_predictor.utils.dataset_analysis import *
from real_estate_predictor.utils.feature_engineering import *
from real_estate_predictor.config.config import PREPROCESSING_PARAMETERS, FEATURE_ENGINEERING_PARAMETERS
class DataCleaner():
    """
    Class to clean the data
    
    Available Functionality:
        - Remove duplicates
        - Handle missing values
        - Remove rows with outliers
        - Standardize text inside categorical columns
    """
    
    def __init__(self, df):
        self.df = df
        
    def remove_duplicates(self, columns, ignore_index = True, strategy = "subset"):
        """
        Remove duplicates from the dataframe based on strategy
        
        Parameters
        ----------
            strategy: str (default = "subset")
                The strategy to use when removing duplicates, available values are "all" and "subset"
            columns: list
                The columns to consider when removing duplicates, only used when strategy is "subset"
            ignore_index: bool
                If True, the index will be reset after removing duplicates
        """
        
        if strategy == "all":
            self.df = self.df.drop_duplicates(ignore_index = ignore_index)
        else:
            self.df = self.df.drop_duplicates(subset = columns, ignore_index = ignore_index)
            
        return self.df
        
    def handle_missing_values(self, strategy, columns = None, threshold = None):
        """
        Removes msising values
        
        Parameters
        ----------
        df : pd.DataFrame
            input pandas dataframe
            
        strategy : str
            Available parameters:
                columns - will completely drop whatever column is inside the `columns` parameter
                rows - will drop rows if there are any missing values in the passed columns
                columns_threshold - will completely drop the column if the number of missing values exceed the passed threshold
                
        columns : list
            default = None
            list of columns used to drop missing values
            
        threshold : float
            default = None
            threshold used to drop columns if the missing values exceed the threshold. Between 0 and 1
            
        Returns
        -------
        
        self.df : pd.DataFrame
        """
        self.df = remove_na_values_by_col(self.df, strategy, columns, threshold)
            
        return self.df
    
    def remove_outliers(self, strategy, columns):
        """
        Remove outliers from the dataframe based on the targeted strategy
        
        Parameters
        ----------
        df : pd.DataFrame
        
        strategy : str
            Default = "all"
            Available strategies:
                all - removes outliers from all numerical columns
                columns - removes outliers from the specified columns in the columns parameter
        
        columns : list
            List of columns to apply outlier removal from
            
        Returns
        -------
        self.df : pd.DataFrame
        
        """
        self.df = removeOutliers(self.df, strategy, columns)
        
        return self.df

    def standardize_text(self, columns = None):
        """
        Given a list of columns, standardize the text inside the columns
        
        Parameters
        ----------
        
        columns : list
            List of columns to standardize the text inside. If this is not given,
            it will automatically use eligible columns from the dataframe self.df
        """
        if not columns:
            columns = self.df.columns
            
        for col in columns:
            if col in PREPROCESSING_PARAMETERS.keys():
                for f in PREPROCESSING_PARAMETERS[col]:
                    self.df = f(self.df)

class Processor():
    """
    Class to do data pre-processing
    
    Available Functionality:
        - Split the data into train and testing sets
        - Impute columns using the Imputer class
        - Encode the categorical values using a specific encoder
        - Transform numerical variables using a specific algorithm
    """
    
    pass

class FeatureEngineering():
    """
    Given a list of columns, standardize the text inside the columns
    
    Parameters
    ----------
    
    columns : list
        List of columns to standardize the text inside. If this is not given,
        it will automatically use eligible columns from the dataframe self.df
    """
    def __init__(self, df):
        self.df = df
        self.features_applied = []
    
    def create_features(self, columns = None, ignore_features = None):
        if not columns:
            columns = self.df.columns
        
        if ignore_features:
            if not isinstance(ignore_features, list):
                raise ValueError("ignore_columns must be a list")
        
        _seen_columns = []
            
        for col in columns:
            if col in FEATURE_ENGINEERING_PARAMETERS.keys():
                for f in FEATURE_ENGINEERING_PARAMETERS[col]:
                    self.df = f(self.df)
    
    def _create_feature(self, 
                        df, 
                        column_dependencies, 
                        feature_function,
                        _seen_columns,
                        ignore_features,
                        ):
        
        if feature_function in self.features_applied:
            return df
        elif not all([True for col in column_dependencies if col in df.columns]):
            raise ValueError(f"Column dependencies not found in the dataframe: {column_dependencies}")
        else:
            try:
                df = feature_function(df)
            except Exception as e:
                raise ValueError(f"Error while creating feature: {e}")
        
        self.features_applied.append(feature_function)
class Trainer():
    """
    Class to train the model
    """
    
    pass