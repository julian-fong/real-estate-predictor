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
    
def XGBoostModel(BaseModel):
    """
    Class to initialize the XGBoost model and fit the model to the dataset
    """
    
    def __init__(self, model = None, param_grid = None, **kwargs):
        self.model = model
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
    
    def select_features(self, strategy = "default", **kwargs):
        pass