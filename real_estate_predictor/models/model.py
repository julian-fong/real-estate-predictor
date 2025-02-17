from copy import deepcopy
from pathlib import Path
import pathlib
import pickle
import datetime as dt
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from xgboost import XGBRegressor, XGBClassifier, plot_importance
import shap

class BaseModel():
    """
    Base class to initialize the model and fit the model to the dataset
    """
    
    def __init__(self):
        raise NotImplementedError("BaseModel is an abstract class and cannot be instantiated")
    
    def fit(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()
    
    def load_model(self, model_path):
        pass
    
    def save_model(self, model_path):
        pass
    
    
class LinearModel(BaseModel):
    """
    Class to initialize the Linear Regression model and fit the model to the dataset
    """
    
    def __init__(self, model):
        self.model = model
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
    
    def load_model(self, model_path):
        pass
        
    def save_model(self, model_path):
        pass
    
def XGBoostRegressor(BaseModel):
    """
    Class to initialize the XGBoost model and fit the model to the dataset
    """
    
    def __init__(self, model = None, param_grid = None, **kwargs):
        """
        3 ways to initialize an XGBoost model:
        - Pass in a model object directly via `model`
        - Pass in a path to a saved model via `model`
        - Pass in the parameters to initialize a new model while leaving `model` = None
        
        self.model_path is the path + the filename of the saved model
        """
        self.model = model
        self._model = self.model if self.model else None
        if isinstance(self._model, str):
            self.model_path = deepcopy(self._model)
            self._model = load_model(self._model)
        else:
            self.model_path = None
        
        if not self._model:
            if not kwargs:
                raise ValueError("No parameters specified for XGBoost model and no model passed")
            self._model = XGBRegressor(**kwargs)
            
        self.param_grid = param_grid
        self.params = kwargs
        
    def set_model_params(self, **params):
        self.params = params
        self._model = XGBRegressor(**params)    
    
    def fit(self, X_train, y_train):
        self.columns = X_train.columns
        self._model.fit(X_train, y_train)
        
    def grid_search(self, X_train, y_train, cv = 5, **kwargs):
        self.columns = X_train.columns
        if not self.param_grid:
            raise ValueError("No parameters specified for grid search")
        
        grid_search = GridSearchCV(self._model, self.param_grid, cv = cv, **kwargs)
        grid_search.fit(X_train, y_train)
        
        #get the results of the grid search via pandas dataframe
        grid_results_df = pd.DataFrame(grid_search.cv_results_)
        grid_results_df.sort_values(by=['rank_test_score']).head()
        
        best_model_params = grid_results_df.sort_values(by=['rank_test_score'])['params'].values[0]
        
        return grid_results_df, best_model_params
        
    def predict(self, X_test):
        """
        should be able to work on pandas Dataframes and new inputs
        """
        return self._model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("r2_val", metrics.r2_score(y_test, y_pred))
        print("mae_val", metrics.mean_absolute_error(y_test, y_pred))
        print("mse_val", metrics.mean_squared_error(y_test, y_pred))
        print("msle_val", metrics.mean_squared_log_error(y_test, y_pred))
    
    def select_columns(self, X_train, y_train, columns):
        """
        Reduces the number of columns of X_train, and y_train to the columns specified
        Will also set self.columns to this new set of columns
        
        """
        assert [True for col in columns if col in self.columns].all(), "Column not found in dataset"
        self.columns = columns
        X_train = X_train[columns]
        y_train = y_train[columns]
        return X_train, y_train
        
    
    def select_features(self, X_train, y_train, strategy = "default", max_features = 25, use_base_model = True,  **kwargs):
        """
        Method to reduce the amount of features buy using feature importance strategies
        
        Available strategies:
        - default: uses the default feature importance from the model via xgboost's `SelectFromModel`
        - shap: uses the SHAP library to calculate feature importance
        
        Parameters
        ----------
        
        strategy: str
            The strategy to use for selecting features, see available strategies above
            
        max_features: int
            The maximum number of features to select
            
        use_base_model: bool
            Whether to use the set model or a new simple model to calculate feature importance
            
        kwargs: dict
            Additional parameters to pass to the feature selection strategy
            
        Returns
        -------
        select_features: object
            The object used to select the features, either a SelectFromModel object or a SHAP explainer object
            
        new_columns: list
            The list of new columns to use for the training and test sets
        
        """
        
        if use_base_model:
            model = XGBRegressor(learning_rate = 0.3)
        else:
            model = self.model
        
        if strategy == "default" or strategy == "xgboost":
            select_features = SelectFromModel(model, max_features = max_features, **kwargs).fit(X_train, y_train)
            feat_index = select_features.get_support()
            #Rename the columns of training and test sets to include column names of top x features
            train_x_xg = pd.DataFrame(X_train, columns = X_train.columns[feat_index])
            new_columns = list(train_x_xg.columns)
            
            return select_features, new_columns
        
        elif strategy == "shap":
            # Create a SHAP explainer
            select_features = shap.Explainer(model).fit(X_train,y_train)
            # Calculate SHAP values for a set of instances
            shap_values = select_features.shap_values(X_train)

            shap_columns = pd.Series(index = X_train.columns, data = np.abs(shap_values[0]))
            train_x_shap = shap_columns.sort_values(ascending = False)[:max_features].index
            new_columns = list(train_x_shap)
            
            return select_features, new_columns
        
        else:
            raise ValueError("Invalid strategy, please use one of the available strategies")
    
    def load_model(self, model_path):
        self.model_path = model_path
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
    
    def save_model(self, model_path = None, filename = None, override = False):
        """
        Saves the xgboost model into a file
        """
        #if we have nothing, try to override using self.model_path
        if not filename and not model_path:
            if override:
                if not self.model_path:
                    raise ValueError("No model path specified in the signture and no model path attribute was set")
                final_path = self.model_path
                with open(final_path, 'wb') as f:
                    pickle.dump(self._model, f)
            else:
                raise ValueError("Parameter override is set to False, please set to true to override previous model")
            
        if not model_path:
            path = pathlib.Path(__file__).parent.parent.absolute().joinpath('storage', 'models')
        else:
            path = Path(path)
            
        if not filename:
            date = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"xgboost_model_{date}.pkl"
        else:
            assert filename.endswith(".pkl"), "Filename must end with .pkl"
        
        path = str(path).replace("\\\\", "\\")+"\\"
        with open(path+filename, "wb") as f:
            pickle.dump(self, f)
                
    
    @classmethod                
    def save(self, path = None, filename = None):
        """
        Save the XBBoostRegressor object to a path
        
        Parameters
        ----------
        
        path : str
            The path to save the FeatureEngineering object
            
        filename : str
            The name of the file to save the object as
        """
        #if path is not specified, put it in the storage/datasets folder
        if not path:
            path = pathlib.Path(__file__).parent.parent.absolute().joinpath('storage', 'processors')
        else:
            path = Path(path)
            
        if not filename:
            date = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"XBBoostRegressor_object_{date}.pkl"
        else:
            assert filename.endswith(".pkl"), "Filename must end with .pkl"
        
        path = str(path).replace("\\\\", "\\")+"\\"
        with open(path+filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod      
    def load(self, path):
        """
        Load the XBBoostRegressor object from a path
        
        Parameters
        ----------
        
        path : str
            The path to load the Processor object
        """
        with open(path, "rb") as f:
            return pickle.load(f)
    
def XGBoostClassifier(BaseModel):
    """
    Class to initialize the XGBoost model and fit the model to the dataset
    """
    
    def __init__(self, model = None, param_grid = None, **kwargs):
        self.model = model
        self._model = self.model if self.model else XGBRegressor(**kwargs)
        self.param_grid = param_grid
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        should be able to work on pandas Dataframes and new inputs
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
    
    def load_model(self, model_path):
        pass
        
    def save_model(self, model_path):
        pass
    
    def select_features(self, X_train, y_train, strategy = "default", max_features = 25, use_base_model = True,  **kwargs):
        """
        Method to reduce the amount of features buy using feature importance strategies
        
        Available strategies:
        - default: uses the default feature importance from the model via xgboost's `SelectFromModel`
        - shap: uses the SHAP library to calculate feature importance
        
        Parameters
        ----------
        
        strategy: str
            The strategy to use for selecting features, see available strategies above
            
        max_features: int
            The maximum number of features to select
            
        use_base_model: bool
            Whether to use the set model or a new simple model to calculate feature importance
            
        kwargs: dict
            Additional parameters to pass to the feature selection strategy
        """
        
        if use_base_model:
            model = XGBRegressor(learning_rate = 0.3)
        else:
            model = self.model
        
        if strategy == "default" or strategy == "xgboost":
            select_features = SelectFromModel(model, max_features = max_features, **kwargs).fit(X_train, y_train)
            feat_index = select_features.get_support()
            #Rename the columns of training and test sets to include column names of top x features
            train_x_xg = pd.DataFrame(X_train, columns = X_train.columns[feat_index])
            xgboost_cols = list(train_x_xg.columns)
            
            return select_features, xgboost_cols
        
        elif strategy == "shap":
            # Create a SHAP explainer
            explainer = shap.Explainer(model).fit(X_train,y_train)
            # Calculate SHAP values for a set of instances
            shap_values = explainer.shap_values(X_train)

            shap_columns = pd.Series(index = X_train.columns, data = np.abs(shap_values[0]))
            train_x_shap = shap_columns.sort_values(ascending = False)[:max_features].index
            shap_cols = list(train_x_shap)
            
            return shap_values, shap_cols
        
        else:
            raise ValueError("Invalid strategy, please use one of the available strategies")