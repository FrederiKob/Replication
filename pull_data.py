import pandas as pd
import numpy as np

#######################################################################################################################
### Load in and Prepare Data // Load in and Prepare Data // Load in and Prepare Data // Load in and Prepare Data // ###
#######################################################################################################################

def pull_data():
        
    ### Pull Data --> Goyal and Welch 2021 Data
    # Source: https://docs.google.com/spreadsheets/d/1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP/edit#gid=2070662242
    ### Pull Data
    data = pd.read_excel("data/Data_Goyal_Welch_2022.xlsx", sheet_name = "Monthly", index_col= "yyyymm")
    data.index = pd.to_datetime(data.index, format = "%Y%m") + pd.DateOffset(months = 1) # end of month values given
    
    # i) dfy -- Default Yield Spread (dfy) is the difference between BAA and AAA-rated corporate bond yields
    data["dfy"] = data["BAA"] - data["AAA"]
    # ii) de -- The Dividend Payout Ratio (d/e) is the difference between the log of dividends and the log of earnings
    data["de"] = np.log(data["D12"]/data["E12"])
    # iii) tms -- the term spread (tms) is the difference between the long term yield on government bonds and the Treasury-bill
    data["tms"] = data["lty"] - data["tbl"]
    # iv) dfr -- Default Return Spread (dfr) is the difference between long-term corporate bond and long-term government bond returns
    data["dfr"] = data.corpr - data.ltr
    # v) dp -- The Dividend Price Ratio (d/p) is the difference between the log of dividends and the log of prices
    data["dp"] = np.log(data["D12"]/data["Index"])
    # vi) dy -- The Dividend Yield (d/y) is the difference between the log of dividends and the log of lagged prices
    data["dy"] = np.log(data.D12/data.Index.shift(1))
    # vii) ep -- Earnings Price Ratio (e/p) is the difference between the log of earnings and the log of prices
    data["ep"] = np.log(data.E12/data.Index)
    # Excess Returns (xr)
    data["xr"] = data.CRSP_SPvw - data.Rfree
    
    data = data.loc[:, ['b/m', 'tbl', 'lty', 'ntis', 'infl', 'ltr', 'svar', 'dfy', 
                               'de', 'tms', 'dfr', 'dp', 'dy', 'ep', 'xr']].dropna()
    
    
    """
    a) Predictors: standardized using an expanding window historical standard deviation (at least 36 months for predictors)
         apply to training and test predictors
    b) Returns: standardized by their trailing 12-month return standard deviation
    """
    
    pred = data.copy().drop("xr",axis=1)
    ret = pd.DataFrame(data.copy().xr)
    
    ## Volatiltiy Standardize
    
    # (i) returns
    ret["roll_std"] = ret.rolling(12).std(ddof=0) # rolling standard deviation 
    ret["xr_std"] = ret.xr/ret.roll_std.shift(1) # standardize xr(t) by std(t-1)
    ret["target"] = ret.xr_std.shift(-1) # pull forward xr so that predictors and target assigned same date
    """ HERE: unsure but I use shifted version of xr """
    ret["lag_1"] = ret.target.shift(1)
    ret_std = ret.drop(["xr_std","roll_std","xr"], axis=1)
    
    # (ii) predictors
    p_std = pred.copy().expanding(36).std(ddof=0)
    """ See Complexity Everywhere --> use up to t-1 std() to standardize t // WHY NOT USE t AS WELL ???"""
    pred_std = pred/p_std.shift(1)
    
    # (iii) combine predictors
    pred_std = pd.concat([pred_std, ret_std], axis=1).dropna()
    ret_std = pred_std.copy().target
    pred_std = pred_std.drop("target",axis=1)
    del(p_std,pred,ret)
    
    return pred_std, ret_std



