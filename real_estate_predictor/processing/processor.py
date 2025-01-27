from real_estate_predictor.utils.dataset_analysis import *
from real_estate_predictor.utils.feature_engineering import *
from real_estate_predictor.config.config import PREPROCESSING_PARAMETERS, FEATURE_ENGINEERING_PARAMETERS

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
class DataCleaner():
    """
    Class to clean the data
    
    Available Functionality:
        - Remove duplicates
        - Handle missing values
        - Remove rows with outliers
        - Standardize text inside categorical columns
        
    Parameters
    ----------
    
    df : pd.DataFrame or string
        The dataframe to clean or a path to the source file of the dataframe.
        Only a path to the source file will enable saving of the DataCleaner object
        to avoid saving the entire dataframe into an object.
    """
    
    def __init__(self, df):
        self.df = df
        self.transformers = []
        
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
                    
    def apply_transformer(self):
        pass
                    
    def save(self, path):
        """
        Save the DataCleaner object to a path
        
        Parameters
        ----------
        
        path : str
            The path to save the DataCleaner object
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path):
        """
        Load the DataCleaner object from a path
        
        Parameters
        ----------
        
        path : str
            The path to load the DataCleaner object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

class Processor():
    """
    Class to construct an sklearn pipeline to do data pre-processing before training
    
    Will construct the preprocessing pipeline for the set of data using sklearn's
    ColumnTransformer and various sklearn pre-processing methods. 
    
    Order of pre-processing:
        - Split the data into train and testing sets
        - Impute numerical columns using `impute_numerical`
        - Transform numerical columns using `transform_numerical`
        - Impute categorical columns using `impute_categorical`
        - Encode categorical columns using `encode_categorical`
    
    Available Functionality:
        - Split the data into train and testing sets
        - Impute columns using the Imputer class
        - Encode the categorical values using a specific encoder
        - Transform numerical variables using a specific algorithm
    """
    
    def __init__(self, df, target_column = None):
        self.df = df
        self.target_column = target_column
        self.numerical_imputer = []
        self.numerical_transformer = []
        self.categorical_imputer = []
        self.categorical_encoder = []
        self.transformers = []
    
    def train_test_split_df(self, target: str, df = None):
        """
        Splits the dataframe into X and y
        
        Parameters
        ----------
        
        target : str
            Mandatory, the target column/variable
            
        df : pd.DataFrame
            Default = None
            If not passed, will use the initialized dataframe
        """
        if not target:
            target = self.target_column
            
        if not df:
            df = self.df
            
        X = df.drop(columns = [target], axis = 1)
        y = df[target]
        
        return X, y
    
    def train_test_split(self, X, y, test_size = None, train_size = None, random_state = None, shuffle = True, stratify = None):
        if not train_size and not test_size:
            train_size = 0.75
            test_size = 0.25
            
        elif not train_size and test_size:
            train_size = 1-test_size
            
        elif train_size and not test_size:
            test_size = 1-train_size
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, test_size = test_size, random_state=random_state,shuffle =  shuffle, stratify = stratify)
        
        return X_train, X_test, y_train, y_test
    
    def impute_numerical(self, df = None, strategy = None, columns = None, f = None):
        """
        Utility to impute numerical columns using a specific strategy
        
        Parameters
        ----------
        
        df : pd.DataFrame
            if not passed, will use the initialized dataframe
            
        strategy : str
            Default = mean
            Available strategies:
                mean - impute the numerical columns via sklearn.SimpleImputer with strategy mean
                median - impute the numerical columns via sklearn.SimpleImputer with strategy median
                knn - impute the numerical columns via sklearn.KNNImputer
        """
        if not df:
            df = self.df
            
        if not columns:
            columns = df.select(include = ["number"])
        
        if not strategy:
            if not f:
                raise ValueError("Either a strategy or a function must be passed")
            else:
                try:
                    strategy = str(f).split("(")[0]
                except:
                    strategy = "CustomFunction"
        else:
            if strategy == "mean":
                from sklearn.impute import SimpleImputer
                f = SimpleImputer(strategy = "mean")
                from sklearn.impute import SimpleImputer
            elif strategy == "median":
                f = SimpleImputer(strategy = "median")
            elif strategy == "knn":
                from sklearn.impute import KNNImputer
                f = KNNImputer(n_neighbors=5)
            
        num = len(self.numerical_imputer)+1

        self.numerical_imputer.append({
            "name": f"num_imputer{num}",
            "pipeline": f, 
            "columns": columns
            })
        
         
    def transform_numerical(self, df = None, strategy = None, columns = None, f = None):
        """
        Utility to transform numerical columns using a specific strategy
        
        Parameters
        ----------
        
        df : pd.DataFrame
        
        strategy : str
            Default = default
            Available strategies:
                default - default the numerical columns via sklearn.StandardScaler
                normalize - normalize the numerical columns via sklearn.Normalizer
                scale - scale the numerical columns via sklearn.MinMaxScaler
                power - apply a power transformation to the numerical columns via sklearn.PowerTransformer
        
        columns : list
        
        """
        if not df:
            df = self.df
            
        if not columns:
            columns = df.select(include = ["number"])
        
        if not strategy:
            if not f:
                raise ValueError("Either a strategy or a function must be passed")
            else:
                try:
                    strategy = str(f).split("(")[0]
                except:
                    strategy = "CustomFunction"
        else:
            if strategy == "default":
                from sklearn.preprocessing import StandardScaler
                f = StandardScaler()
            elif strategy == "normalize":
                from sklearn.preprocessing import Normalizer
                f = Normalizer()
            elif strategy == "scale":
                from sklearn.preprocessing import MinMaxScaler
                f = MinMaxScaler()
            elif strategy == "power":
                from sklearn.preprocessing import PowerTransformer
                f = PowerTransformer()
            else:
                raise ValueError(f"Invalid strategy, found {strategy}")
            
        num = len(self.numerical_transformer)+1
        
        self.numerical_transformer.append({
            "name": f"num_transformer_{num}",
            "pipeline": f, 
            "columns": columns
        })
        
    def impute_categorical(self, df = None, strategy = None, columns = None, f = None):
        """
        Utility to impute categorical columns using a specific strategy
        
        Parameters
        ----------
        
        df : pd.DataFrame
        
        strategy : str
            Default = most_frequent
            Available strategies:
                most_frequent - impute the categorical columns via sklearn.SimpleImputer with strategy most_frequent
        
        """
        if not columns:
            columns = df.select(include = ["object"])
        
        if not strategy:
            if not f:
                raise ValueError("Either a strategy or a function must be passed")
            else:
                try:
                    strategy = str(f).split("(")[0]
                except:
                    strategy = "CustomFunction"
        else:
            if strategy == "most_frequent":
                from sklearn.impute import SimpleImputer
                f = SimpleImputer(strategy = "most_frequent")
            else:
                raise ValueError(f"Invalid strategy, found {strategy}")
        
        num = len(self.categorical_imputer)+1
        
        self.categorical_imputer.append({
            "name": f"cate_imputer_{num}",
            "pipeline": f, 
            "columns": columns
            })
            
    def encode_categorical(self, df = None, strategy = None, columns = None, f = None):
        """
        Utility to encode categorical columns using a specific strategy
        
        Parameters
        ----------
        
        df : pd.DataFrame
        
        strategy : str
            Default = One Hot Encoding
            Available strategies:
                onehot - encode the categorical columns via sklearn.OneHotEncoder
                label - encode the categorical columns via sklearn.LabelEncoder
                
        """
        if not columns:
            columns = df.select(include = ["object"])
        
        if not strategy:
            if not f:
                raise ValueError("Either a strategy or a function must be passed")
            else:
                try:
                    strategy = str(f).split("(")[0]
                except:
                    strategy = "CustomFunction"
        else:
            if strategy == "onehot":
                #set sparse_output to False to return pandas dataframe
                from sklearn.preprocessing import OneHotEncoder
                f = OneHotEncoder(sparse_output=False)
            elif strategy == "label":
                from sklearn.preprocessing import LabelEncoder
                f = LabelEncoder()
            else:
                raise ValueError(f"Invalid strategy, found {strategy}")
        
        num = len(self.categorical_encoder)+1
        self.categorical_encoder.append({
            "name": f"cate_encoder_{num}",
            "pipeline": f, 
            "columns": columns
            })
        
    def apply_transformer(self):
        self._construct_numerical_pipeline()
        self._construct_categorical_pipeline()
        self.preprocessor = ColumnTransformer(transformers = self.transformers, remainder = "passthrough")
    
    def _construct_numerical_pipeline(self):
        seen_columns = []
        #if a imputed column also exists in the transformer columns, combine them into one pipeline
        for numerical_imputer in self.numerical_imputer:
            #first loop through the imputer tuples
            for col in numerical_imputer["columns"]:
                #at the minimum, initialize a new imputation pipeline for the column
                transformer = Pipeline(steps = [])
                #add the imputation function to the pipeline
                transformer.steps.append(
                        (f"numerical_imputer_{col}", numerical_imputer["pipeline"])
                    )
                for numerical_transformer in self.numerical_transformer:
                    if col in numerical_transformer["columns"]:
                        #add it to seen columns to avoid duplicates
                        seen_columns.append(col)
                        transformer.steps.append(
                                (f"numerical_transformer_{col}", numerical_transformer["pipeline"])
                        )
                self.transformers.append((f"transformer_{len(self.transformers)+1}", transformer, [col]))
                
        #add in the remaining transformers inside numerical_transformer that were not seen in imputer
        for numerical_transformer in self.numerical_transformer:
            if not self._check_if_subset(numerical_transformer["columns"], seen_columns):
                transformer = Pipeline(steps = [])
                transformer.steps.append(
                    (f"numerical_transformer_{str(numerical_transformer['columns'])}", numerical_transformer["pipeline"])
                )
                self.transformers.append((f"transformer_{len(self.transformers)+1}", transformer, numerical_transformer["columns"]))
    
    def _construct_categorical_pipeline(self):
        seen_columns = []
        #if a imputed column also exists in the transformer columns, combine them into one pipeline
        for categorical_imputer in self.categorical_imputer:
            #first loop through the imputer tuples
            for col in categorical_imputer["columns"]:
                #at the minimum, initialize a new imputation pipeline for the column
                transformer = Pipeline(steps = [])
                #add the imputation function to the pipeline
                transformer.steps.append(
                    ("categorical_imputer", categorical_imputer["pipeline"])
                )
                for categorical_encoder in self.categorical_encoder:
                    if col in categorical_encoder["columns"]:
                        #add it to seen columns to avoid duplicates
                        seen_columns.append(col)
                        transformer.steps.append(
                            ("categorical_encoder", categorical_encoder["pipeline"])
                        )
                self.transformers.append((f"transformer_{len(self.transformers)+1}", transformer, [col]))

        #add in the remaining transformers inside numerical_transformer that were not seen in imputer
        for categorical_encoder in self.categorical_encoder:
            if not self._check_if_subset(categorical_encoder["columns"], seen_columns):
                transformer = Pipeline(steps = [])
                transformer.steps.append(
                    ("categorical_encoder", categorical_encoder["pipeline"])
                )
                self.transformers.append((f"transformer_{len(self.transformers)+1}", transformer, categorical_encoder["columns"]))
    
    def _check_if_subset(self, subset, larger_set):
        return all([True if col in larger_set else False for col in subset])
    
    def fit(self, X):
        if not self.preprocessor:
            raise ValueError("ColumnTransformer not initialized, please run `apply_column_transformer`")
        
        self.preprocessor.set_output(transform="pandas")
        self.preprocessor.fit(X)
        
    def transform(self, X):
        if not self.preprocessor:
            raise ValueError("ColumnTransformer not initialized, please run `apply_column_transformer`")

        feature_names = []
        for processor in self.preprocessor.named_transformers_.keys():
            if isinstance(self.preprocessor.named_transformers_[processor], Pipeline):
                #get all the names of the transformers
                categorical_names = [t[0] for t in self.preprocessor.named_transformers_[processor].steps]
                if "categorical_encoder" in categorical_names:
                    feature_names += self.preprocessor.named_transformers_[processor].get_feature_names_out().tolist()
                else:
                    feature_names += self.preprocessor.named_transformers_[processor].feature_names_in_.tolist()
            else:
                feature_names += self.preprocessor.named_transformers_[processor].feature_names_in_.tolist()
        
        X_transformed = self.preprocessor.transform(X)
        X_transformed.columns = feature_names
        
        return X_transformed
    
    def fit_transform(self, X):
        if not self.preprocessor:
            raise ValueError("ColumnTransformer not initialized, please run `apply_column_transformer`")
        
        feature_names = []
        self.preprocessor.set_output(transform="pandas")
        X_transformed = self.preprocessor.fit_transform(X)
        for processor in self.preprocessor.named_transformers_.keys():
            if isinstance(self.preprocessor.named_transformers_[processor], Pipeline):
                #get all the names of the transformers
                categorical_names = [t[0] for t in self.preprocessor.named_transformers_[processor].steps]
                if "categorical_encoder" in categorical_names:
                    feature_names += self.preprocessor.named_transformers_[processor].get_feature_names_out().tolist()
                else:
                    feature_names += self.preprocessor.named_transformers_[processor].feature_names_in_.tolist()
            else:
                feature_names += self.preprocessor.named_transformers_[processor].feature_names_in_.tolist()                
        X_transformed.columns = feature_names

        return X_transformed
        
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
        self.column_cache = {}
        self.transformer = []
    
    def create_features_old(self, columns = None, ignore_features = None):
        if not columns:
            columns = self.df.columns
        
        if ignore_features:
            if not isinstance(ignore_features, list):
                raise ValueError("ignore_columns must be a list")
        
        _seen_columns = []
        
        for feature in FEATURE_ENGINEERING_PARAMETERS.keys():
            if feature in columns and feature not in ignore_features:
                column_dependencies, f = FEATURE_ENGINEERING_PARAMETERS[feature]
                
                if not all([True if col in self.df.columns else False for col in column_dependencies]):
                    pass
                else:
                    self.df = f(self.df)
                    self.features_applied.append((feature, f))
    
    # TODO: Implement a way to check if the feature is already in the cache        
    def create_features(self, columns = None, ignore_features = None, feature = None):
        if not columns:
            columns = self.df.columns
            
        for feature in TEST_FEATURE_ENGINEERING_PARAMETERS_1:
            is_feature_in_cache = self._check_feature_in_cache(feature, self.column_cache)
            
            if not is_feature_in_cache:
                for col in TEST_FEATURE_ENGINEERING_PARAMETERS_1[feature][0]:
                    if not col:
                        self.column_cache[feature] = True
                    else:
                        self.create_features()
                        
                    
    def _check_feature_in_cache(self, feature, column_cache):
        """
        Helper that checks the feature
        """
        #if feature is inside the cache, return False
        if feature in column_cache.keys():
            return True
        
        return False

    def _evaluate_feature(self, list_of_dependencies):
        if not all([True if col in self.df.columns else False for col in list_of_dependencies]):
            return True
        else:
            return False

        # TODO: Figure out a way to attempt to create missing column dependencies if the dependency is missing
        # and is inside the FEATURE_ENGINEERING_PARAMETERS dictionary
                        
    def apply_transformer(self):
        pass
class Trainer():
    """
    Class to train the model
    """
    
    pass