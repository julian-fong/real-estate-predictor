import pandas as pd
import numpy as np
import os
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgbm

from scripts.feature_engineering import feature_engineering

def create_model(days, type_ = None):
    data = feature_engineering(days, type_)

    x, y = data.drop(['soldPrice'], axis = 1), data['soldPrice']

    x = x.rename(columns = lambda z: re.sub('[^A-Za-z0-9_]+', '', z))

    #In the case we have duplicate columns, just remove them both
    if list(x.columns) != set(x.columns):
        col_to_drop = set([col for col in x if list(x.columns).count(col) > 1])
        x = x.drop(col_to_drop, axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

    # print('Shape of x_train: ' + str(x_train.shape))
    # print('Shape of y_train: ' + str(y_train.shape))
    # print('Shape of x_test: ' + str(x_test.shape))
    # print('Shape of y_test: ' + str(y_test.shape))
    
    #Define XGBoost parameters tree_method = 'gpu_hist'
    base_regressor = lgbm.LGBMRegressor(learning_rate = 0.3,random_state=0)

    #Get top 100 features using XGBoost
    select_feat = SelectFromModel(base_regressor,threshold=-np.inf,max_features=25).fit(x_train,y_train)
    #Get indices of top 100 features 
    feat_index = select_feat.get_support()

    #Rename the columns of training and test sets to include column names of top 100 features
    train_x = pd.DataFrame(x_train, columns = x_train.columns[feat_index])
    test_x = pd.DataFrame(x_test, columns= x_test.columns[feat_index])
    
    #LightGBM parameters
    params = {
            'learning_rate': [0.1, 0.2],
            #'subsample': [0.25, 0.5, 1.0],
            #'max_depth': [5, 7, 9,],
            #'num_leaves': [15, 30, 45],
            #'min_child_samples': [10,20,30],
            }
    
    #Define XGBoost base model 
    lgbm_model = lgbm.LGBMRegressor(n_estimators=100, device = 'gpu')

    #Obtain the best set of parameters
    grid_search = GridSearchCV(lgbm_model, params, cv=5,verbose=3)
    grid_search.fit(train_x, y_train)


    #Get results of gridsearch 
    grid_results_df = pd.DataFrame(grid_search.cv_results_)
    grid_results_df.sort_values(by=['rank_test_score']).head()

    #Apply the best parameters as our best model
    best_model = lgbm.LGBMRegressor(n_estimators=100, device = 'gpu')
    best_model.set_params(**grid_results_df.sort_values(by=['rank_test_score'])['params'].values[0])
    best_model = best_model.fit(train_x, y_train)

    print(grid_results_df.sort_values(by=['rank_test_score'])['params'].values[0])
    #Predict the test set values
    y_pred = best_model.predict(test_x)

    print(y_pred)

    #Calculate our performance metrics
    r2 = 100*metrics.r2_score(y_test, y_pred)
    mae = 100*metrics.mean_absolute_error(y_test, y_pred)

    #Save the best model along with the feature engineered columns and the indexes of our top features
    try:
        print("Saving best model parameters...")
        model_parameters = {"model": best_model, "columns": x_train.columns, "idx": feat_index}
        filename = f"{type_}_model_parameters.sav"
        with open(os.getcwd()+f"\\models\\{filename}", 'wb') as f:
            pickle.dump(model_parameters, open(os.getcwd()+f"\\models\\{filename}", 'wb'))
        
        print(f"Best model paramters for type {type_} successful")
    except:
        print("best model did not save successfully")

    return r2, mae


def main(days, type_):
    if type_:
        print(f"Creating {type_} model...")
        r2, mae = create_model(days, type_)

        print(f"R^2 Score for type {type_}: {r2}")
        print(f"MAE Score for type {type_}: {mae}")
    else:
        print("Creating sale model...")
        sale_r2, sale_mae = create_model(days, 'sale')

        print("Creating lease model...")
        lease_r2, lease_mae = create_model(days, 'lease')

        print(f"R^2 Score for type sale: {sale_r2}")
        print(f"MAE Score for type sale: {sale_mae}")

        print(f"R^2 Score for type lease: {lease_r2}")
        print(f"MAE Score for type lease: {lease_mae}")

