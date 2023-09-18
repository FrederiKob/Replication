import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score


class run_backtest():
    ############
    def __init__(self, rff_master = None, use_type=np.float32):
        # pull from rff_master
        self.scaling = rff_master.scaling
        self.rng_new = rff_master.rng_new
        self.vol_stand = rff_master.vol_stand
        self.alpha_adjust = rff_master.alpha_adjust
        self.GoyWel = rff_master.GoyWel
        # X and Y
        self.Y = rff_master.returns
        if self.GoyWel:
            self.X = rff_master.predictors
            self.P = rff_master.P
        else:
            self.X = rff_master.signals
            self.P = rff_master.P
        # as_type --> speed at the cost of minimal accuracy
        self.use_type = np.float32
            
    ########### 
    def predict(self, alpha = None, T = None):
        # add alpha and T
        self.alpha = alpha
        self.T = T
        
        backtest = []
        T_max, self.P = self.X.shape
        self.c = self.P / self.T

        index = list(range(self.T, T_max))
        for t in index:
            if self.vol_stand:
                x_train = self.X[t-self.T:t].astype(self.use_type)
                x_std = x_train.std(ddof=0,axis=0)
                x_train = x_train/x_std
                y_train = self.Y[t-self.T:t].astype(self.use_type)

                x_test = self.X[t:t+1].astype(self.use_type)/x_std
                y_test = self.Y[t:t+1].astype(self.use_type)
                
            else:
                x_train = self.X[t-self.T:t].astype(self.use_type)
                y_train = self.Y[t-self.T:t].astype(self.use_type)

                x_test = self.X[t:t+1].astype(self.use_type)
                y_test = self.Y[t:t+1].astype(self.use_type)

            # Ridge.alpha is adjusted by T to get the same results as in the paper. 
            if self.alpha_adjust:
                beta = Ridge(alpha=(self.alpha*self.T), solver="svd", fit_intercept=False, normalize=False).fit(x_train, y_train).coef_
            else:
                beta = Ridge(alpha=(self.alpha), solver="svd", fit_intercept=False, normalize=False).fit(x_train, y_train).coef_
            forecast = x_test @ beta
            # Keep in mind that R_test is 1-step ahead, thus it satisfies beta'*S_t*R_t+1
            timing_strategy = forecast * y_test

            backtest.append({
                "index": y_test.index[0],
                "beta_norm": np.sqrt((beta**2).sum()),
                "forecast": forecast[0],
                "timing_strategy": timing_strategy[0],
                "market_return": y_test[0]
            })
        # The last value for market_return is NaN since it is predicting the next month
        self.backtest = pd.DataFrame(backtest).set_index("index")
        self.prediction = self.backtest["forecast"]
        return self


    def performance(self, time_factor:int = 12):
        """Calculates various performance metrics for the backtest.

        Args:
            time_factor (int, optional): Factor to annualize the data e.g. 12 for a monthly frequency. Defaults to 12.
        """
        data = self.backtest.dropna()
        # Calculate Alpha & Beta of the timing strategy
        market_reg = LinearRegression().fit(data[["market_return"]], data["timing_strategy"])
        beta = market_reg.coef_[0]
        alpha = market_reg.intercept_

        # Annualize returns
        sqrt_time_factor = np.sqrt(time_factor)
        mean = data["timing_strategy"].mean()*time_factor
        std = data["timing_strategy"].std()*sqrt_time_factor
        mean_market = data["market_return"].mean()*time_factor

        self.performance = {
            "beta_norm_mean" : data["beta_norm"].mean(),
            "Market Sharpe Ratio" : (data["market_return"].mean()*time_factor) / (data["market_return"].std()*sqrt_time_factor),
            "Expected Return" : mean,
            "Volatility" : std,
            "R2" : r2_score(data["market_return"], data["forecast"]),
            "SR" : mean/std,
            # Adjust IR by beta to get a better scaling since mean converges to zero for higher c
            "IR" : (mean - mean_market*beta)/std, 
            "Alpha" : alpha, 
            # Does sign(forecast) match sign(market_return)?
            "Precision": precision_score(data["market_return"] > 0, data["forecast"] > 0),
            "Recall": recall_score(data["market_return"] > 0, data["forecast"] > 0),
            "Accuracy": accuracy_score(data["market_return"] > 0, data["forecast"] > 0),
        }
        return self




