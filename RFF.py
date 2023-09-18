import numpy as np
import pandas as pd
import math

class rff():
    # define self --> use update to write P seed and signals to self
    def __init__(self, predictors = None, returns = None, end = "2021-01-01", 
                 scaling: bool = False, rng_new: bool = True, vol_stand: bool = None,
                 alpha_adjust: bool = True, GoyWel: bool = False):
        # reduce to relevant 
        idx_end = np.where(predictors.index == end)[0][0]+1
        self.end = end
        self.predictors = predictors.iloc[:idx_end,:]
        self.returns = returns.iloc[:idx_end]
        # general booleans
        self.scaling = scaling
        self.rng_new = rng_new
        self.vol_stand = vol_stand
        self.alpha_adjust = alpha_adjust
        self.GoyWel = GoyWel
        # parameters -> defined later
        self.P = None
        self.seed = None
        self.signals = None
    

    def generateRFF(self): #-> np.ndarray:
        ## (i) set seed
        if self.seed is None:
            raise ValueError("No specific seed was given!")
        np.random.seed(self.seed)
        
        ## (ii) random draws
        num_features = self.predictors.shape[1]
        num_rand = int(math.ceil(self.P/2)) # for odd numbers --> later drop one signal at random
        
        if self.rng_new: 
            w_i = np.random.randn(num_features, num_rand)
        else:
            w_i = np.random.multivariate_normal(mean=np.array([0]*num_features), cov=np.identity(num_features),
                                                size=num_rand).T

        ## (iii) calculate signals
        if self.scaling:
            sin = self.P**(-0.5) * np.sin(2 * self.predictors @ w_i)
            cos = self.P**(-0.5) * np.cos(2 * self.predictors @ w_i)
        else:
            sin = np.sin(2 * self.predictors @ w_i)
            cos = np.sin(2 * self.predictors @ w_i)
        signals = np.concatenate([sin, cos], axis=1)
        
        if self.P % 2 != 0:
            # random item to drop --> due to one signal too many (previously math.ceil increased by one)
            drop = np.random.randint(0,self.P+1)
            # drop random element --> given np.random it respects the seed previously set
            signals = np.delete(signals,drop,axis=1)    
            
        ## (iv) return RFFs (signals)
        return signals
        
    def update(self, generateRFF=generateRFF, seed = None, P = None):
        self.seed = seed
        if self.GoyWel and P != 15:
            raise ValueError("You are using Goyal Welch 15 parameter setting, wrong P input!")
        self.P = P
        self.signals = generateRFF(self)



