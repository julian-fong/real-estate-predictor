from sklearn.base import BaseEstimator

class Imputer(BaseEstimator):
    """
    Class for imputation strategies
    
    
    """
    pass


class Encoder(BaseEstimator):
    """
    Class for encoding strategies
    
    
    """
    pass

class NumericalTransformer(BaseEstimator):
    """
    Class for numerical transformation strategies
    
    
    """
    pass

#iterating through al nodes in the graph

#empty cache {} to store the results of the nodes
# cache = {}

#start with sqft
#look at sqft, and check its dependencies, 
#all dependencies are satisfied since it is []

# for each dependency in sqft, if the dependency is in the cache and False

#sqft sincei ts true, it is added to the cache as True.
#effectively it can be used in the model as also can now be u sed as a node as a dependency

#next node is ppsqft
#check if ppsqft is in the cache, it is not
#recursively call the function on the dependencies of ppsqft since its not in the cache

    #need to go dependency by dependency
    #check sqft_avg, check if its in the cache, it is not
    #recursively call the function on the dependencies of sqft_avg since its not in the cache

        #need to go dependency by dependency
        #check sqft, check if its in the cache, it is in the cache and its True
        #sqft_avg is then added to the cache as True, can be used in the model and set as a node
        
        #check listPrice, check if its in the cache, it is not
        #listPrice isn't in the cache, and its also not in the key dictionary
        #listPrice is added to the cache as False, can't be used in the model
        
    #since not all of the dependencies are true, ppsqft is added to the cache as False, can't be used in the model
    
#iterate through the cache, find everything that is true
#in this case its sqft and sqft_avg, 