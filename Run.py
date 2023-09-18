import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import itertools
from joblib import Parallel, delayed

### own
from pull_data import pull_data

### pull standardized data 
pred_std, ret_std = pull_data()

### define parameters
T_grid = [12,60]
max_iterations = 10
alpha_grid = [1000]
P_grid = [12000]

iterations = list(range(1,max_iterations))

from Backtest import run_backtest
from RFF import rff
rff_sig = rff(pred_std, ret_std, scaling=True, rng_new=True, vol_stand=True, alpha_adjust=True, GoyWel=False)
def simulation(rff_sig, T, P, alpha, iteration):
    c = P/T
    # update master
    rff_sig.update(seed=iteration, P=P)
    # 1-step ahead prediction
    res_iter =  run_backtest(rff_sig).predict(alpha=alpha, T=T).performance()
    prediction = res_iter.backtest
    performance = res_iter.performance
    performance.update({"P": P, "a": alpha, "c": c, "T": T, "iteration": iteration,
                        "beta_norm":prediction.beta_norm, "forecast":prediction.forecast,
                        "timing_strategy":prediction.timing_strategy, "market_return":prediction.market_return})    
    return performance

results_all = Parallel(n_jobs=5, backend="multiprocessing")(
    delayed(simulation)(rff_sig,T,P,alpha,iteration=seed) 
    for T, P, alpha, seed in itertools.product([12], P_grid, alpha_grid, iterations)
)




metrics = pd.DataFrame(results_all.copy())
metric = metrics.drop(metrics.columns[-4:], axis=1)
metrics.to_parquet((f"data/metrics.parquet"))

metrics_mean = metrics.groupby(["P", "a", "c", "T"]).mean().reset_index().drop("iteration", axis=1)
metrics_mean["log10(z)"] = np.log10(metrics_mean["a"])

## write all results
names = ["beta_norm","forecast","market_return","timing_strategy"]
combs = list(itertools.product(P_values, alpha_grid))
sub_dic = {i:[] for i in names}
results = {i:sub_dic for i in combs}

df = pd.DataFrame(results_all.copy())
for c in combs:
    _df = df[(df.P == c[0]) & (df.a == c[1])]
    for ele in names:
        _conc = pd.concat([_df[ele].iloc[i] for i in iterations], axis=1)
        _conc.columns = _df.iteration
        results[c][ele] = _conc 


################
### Save Results
from Functions_Pull_Save import save_backup
objects_file = [metric.copy(), metrics_mean.copy(), results.copy()]
objects_name = ["metric", "metrics_mean", "results"]
save_backup(objects_file = objects_file, objects_name = objects_name)



#############################################################################################################
#-----------------------------------------------------------------------------------------------------------#
#############################################################################################################

### Baseline
# Do the same 1-step ahead prediction with the dataset instead of RFF and pick the best metric for each possible z
baseline = []

# Compute baseline performance for each z
for a in alpha_grid:
    base = Backtest(rff_master, use_signals=False, vol_stand=False, alpha_adjust=False).predict(alpha = a, T=T).performance()
    performance = base.performance   
    performance.update({"a":a})
    baseline.append(performance)    
    

baseline = pd.DataFrame(baseline)
baseline_dict = baseline[["Expected Return", "SR", "IR", "Alpha", "Precision", "Recall", "Accuracy"]].max().to_dict()

print("Baseline Results:")
for k,v in baseline_dict.items():
    print(f"\t - {k}: {v}")
    
    
##########
### Result

"""
Paper (Note: Footnote[25])

More specifically, the first column reports summary statistics for the market return with rolling 12-month volatility 
standardization. Thus, the buy-and-hold version of this asset is itself a basic timing strategy, where timing is inversely 
proportional to rolling volatility. We do this simply because the standardized market is the target in our forecasting analysis. 
Our results across the board are generally insensitive to, and our conclusions entirely unaffected by, whether we work with the 
raw or volatility standardized market return. As noted earlier, we prefer to use the volatility standardized market because it 
aligns more directly with our theoretical framework.
"""

# Page 48 reports R2 as "per month". The Sharpe Ratio value seems to be more fitting inf you annualize the data
result = metrics_mean[(metrics_mean.a==1000) & (metrics_mean.c == 1000)][["Expected Return", "R2", "SR", "IR"]].max().to_dict()

print("Our Results:")
for k,v in result.items():
    print(f"\t - {k}: {v}")
    
    
#########
### Plots

"""
Here we present several visualizations to analyze our metrics. We use a logarithmic scale on the left to better represent 
the results for each c in the range [0, 1000]. The right-hand side provides plots similar to those in the original paper, 
but with a broken x-axis for c in the ranges [0, 50] and [950, 1000].

We also provide plots for specific metrics, such as the Sharpe Ratio, for particular ranges of log10(z). This allows us to 
better visualize the differences for each value of log10(z). Additionally, we show plots for various gamma values to validate 
footnote [24]:

    We set γ = 2. Our results are generally insensitive to γ, as discussed in the robustness section below.

The critical point at c=1 (meaning the number of observations equals the number of features resulting in) is indicated by a 
vertical grey line. The baseline value of a simple Ridge regression without Random Fourier Features is marked by a horizontal 
black line, aiding in the comparison and identification of improvements.


Overall, our plots show a trend consistent with the original paper, although there are differences in specific values. 
These discrepancies may stem from different specific implementation choices (e.g., beta adjustment?).

The metrics are displayed blow. Compared to page 39:

beta_norm_mean: comparable
Expected Return: 10x higher. Figure 7 of the paper indicates that it is the "Market Timing Performance", which is what I used (annualized)
Volatility: our results are approximately twice as high
R2: comparable
The following metrics are extremely similar to those shown in Figure 8 of the paper (p41)

Sharpe Ratio
Information Ratio
Alpha

All values are below the regression benchmark for gamma==2
"""
# https://stackoverflow.com/a/34934631/3002299


result = metrics_mean

fig = plt.figure(figsize=(20, 40))
outer = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.1)

plot_cols = ["beta_norm_mean", "Expected Return", "Volatility", "R2", "SR", "IR", "Alpha", "Precision", "Recall", "Accuracy"]
for i in range(2): # Outer
    inner = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.5)
    if i == 0:
        for j in range(7): # Inner, left column
            col = plot_cols[j]
            ax = plt.Subplot(fig, inner[j])
            result.set_index("c").groupby("log10(z)")[col].plot(ax=ax, title=col)
            ax.axvline(1, c="grey", linestyle="--") # Line vor a complexity of 1, eaning 12 features and 12 data points
            if col in baseline_dict.keys():
                ax.axhline(baseline_dict[col], c="blacK", linestyle="--") 
            ax.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
            ax.set_xlabel("c, log scale")
            ax.set_xscale('log')
            fig.add_subplot(ax)
    else:  
        for j in range(10): # Inner, right column
            col = plot_cols[j]
            # Two plots with a cut axis
            double_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[j], wspace=0.1, hspace=0.1)
            
            ax1 = plt.Subplot(fig, double_inner[0])
            ax1.axvline(1, c="grey", linestyle="--") # Line vor a complexity of 1, eaning 12 features and 12 data points
            ax2 = plt.Subplot(fig, double_inner[1], sharey = ax1)
            if col in baseline_dict.keys():
                ax1.axhline(baseline_dict[col], c="blacK", linestyle="--") 
                ax2.axhline(baseline_dict[col], c="blacK", linestyle="--") 
            result.set_index("c").groupby("log10(z)")[col].plot(ax=ax1, title=col)
            result.set_index("c").groupby("log10(z)")[col].plot(ax=ax2)
            ax1.set_xlim(0, 50)
            ax2.set_xlim(950, 1000)

            ax1.axvline(x=50, linestyle="--", c="black")
            ax2.axvline(x=950, linestyle="--", c="black")

            # hide the spines between ax and ax2
            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            #ax1.yaxis.tick_left()
            #ax1.tick_params(labelright='off')
            ax2.yaxis.tick_right()

            d = .015 # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d,1+d), (-d,+d), **kwargs)
            ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d,+d), (1-d,1+d), **kwargs)
            ax2.plot((-d,+d), (-d,+d), **kwargs)

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
plt.savefig(f"plots/metrics_{T}.jpg", bbox_inches='tight', dpi=1600)
print(f"T == {T}")
plt.show()




gamma = 2

for gamma in [0.5, 2]:
    result = metrics_mean[metrics_mean["gamma"]==gamma]

    fig = plt.figure(figsize=(20, 40))
    outer = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.1)

    plot_cols = ["beta_norm_mean", "Expected Return", "Volatility", "R2", "SR", "IR", "Alpha", "Precision", "Recall", "Accuracy"]
    for i in range(2): # Outer
        inner = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.5)
        if i == 0:
            for j in range(10): # Inner, left column
                col = plot_cols[j]
                ax = plt.Subplot(fig, inner[j])
                result.set_index("c").groupby("log10(z)")[col].plot(ax=ax, title=col)
                ax.axvline(1, c="grey", linestyle="--") # Line vor a complexity of 1, eaning 12 features and 12 data points
                if col in baseline_dict.keys():
                    ax.axhline(baseline_dict[col], c="blacK", linestyle="--") 
                ax.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
                ax.set_xlabel("c, log scale")
                ax.set_xscale('log')
                fig.add_subplot(ax)
        else:  
            for j in range(10): # Inner, right column
                col = plot_cols[j]
                # Two plots with a cut axis
                double_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[j], wspace=0.1, hspace=0.1)
                
                ax1 = plt.Subplot(fig, double_inner[0])
                ax1.axvline(1, c="grey", linestyle="--") # Line vor a complexity of 1, eaning 12 features and 12 data points
                ax2 = plt.Subplot(fig, double_inner[1], sharey = ax1)
                if col in baseline_dict.keys():
                    ax1.axhline(baseline_dict[col], c="blacK", linestyle="--") 
                    ax2.axhline(baseline_dict[col], c="blacK", linestyle="--") 
                result.set_index("c").groupby("log10(z)")[col].plot(ax=ax1, title=col)
                result.set_index("c").groupby("log10(z)")[col].plot(ax=ax2)
                ax1.set_xlim(0, 50)
                ax2.set_xlim(950, 1000)

                ax1.axvline(x=50, linestyle="--", c="black")
                ax2.axvline(x=950, linestyle="--", c="black")

                # hide the spines between ax and ax2
                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                #ax1.yaxis.tick_left()
                #ax1.tick_params(labelright='off')
                ax2.yaxis.tick_right()

                d = .015 # how big to make the diagonal lines in axes coordinates
                # arguments to pass plot, just so we don't keep repeating them
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d,1+d), (-d,+d), **kwargs)
                ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

                kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                ax2.plot((-d,+d), (1-d,1+d), **kwargs)
                ax2.plot((-d,+d), (-d,+d), **kwargs)

                fig.add_subplot(ax1)
                fig.add_subplot(ax2)
    plt.savefig(f"plots/metrics_{gamma}.jpg", bbox_inches='tight')
    print(f"gamma == {gamma}")
    plt.show()



result.set_index("c").groupby("log10(z)")["R2"].plot(ylim=(-0.2,0.01), figsize=(10,5), title="R2")
plt.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
plt.show()

result.set_index("c").groupby("log10(z)")["SR"].plot(ylim=(0.0, 0.5), figsize=(10,5), title="SR")
plt.gca().axhline(baseline_dict["SR"], c="blacK", linestyle="--") 
plt.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
plt.show()

result.set_index("c").groupby("log10(z)")["IR"].plot(ylim=(0.05,0.25), figsize=(10,5), title="IR")
plt.gca().axhline(baseline_dict["IR"], c="blacK", linestyle="--") 
plt.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
plt.show()

result.set_index("c").groupby("log10(z)")["Alpha"].plot(ylim=(0.00,0.02), figsize=(10,5), title="Alpha")
plt.gca().axhline(baseline_dict["Alpha"], c="blacK", linestyle="--") 
plt.legend(loc="upper left", title="log10(z)", bbox_to_anchor=(1, 1.05))
plt.show()


##############################################
### Market Timing positions vs NBER Recessions



col ="forecast"

# Create RFF model and use it to predict returns
rff_data = RFF(n=6000, gamma=2).features(data)
backtest = Backtest(z=1000, T=12).predict(X=rff_data, y=returns.shift(-1))
backtest = backtest.backtest

# Prepare data for plotting
plot_data = pd.DataFrame()
plot_data[col] = backtest[col]
plot_data["6m MA"] = plot_data[col].rolling(6).mean()

# Get recessions dates from NBER
recessions = [t for date_list in nber.apply(lambda x: pd.date_range(x["peak"], x["trough"]), axis=1).values for t in date_list]
plot_data["NBER Recession"] = plot_data.index.isin(recessions).astype(int)

plot_data = plot_data.dropna()


# Create plot
fig, ax = plt.subplots(figsize=(15,7))
plot_data[col].plot(ax=ax, alpha=0.7, c="lightblue")
plot_data["6m MA"].plot(ax=ax, c="steelblue")
ax.set_ylim(plot_data["6m MA"].min()*1.1, plot_data["6m MA"].max()*1.1)

# Overlay recession periods
ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], 
                where=plot_data["NBER Recession"] == 1 ,color='grey', alpha=0.3,  label="NBER Recession")

ax.legend(loc="upper right")
ax.set_title(f"Column: {col}")
plt.tight_layout()
plt.savefig(f"plots/result_{col}.jpg")
plt.show()














