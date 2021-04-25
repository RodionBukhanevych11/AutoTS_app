import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf,q_stat, adfuller
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from scipy.stats import spearmanr, pearsonr,probplot, moment
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from collections import Counter
from itertools import chain
import scipy.stats as stats
from fracdiff import Fracdiff, FracdiffStat, fdiff
import warnings
warnings.simplefilter('ignore')


def read_df(filename):
    try:
        if filename[-4:]=='xlsx':
            df = pd.read_excel(filename, index_col=None)
        elif filename[-3:]=='csv':
            df = pd.read_csv(filename, index_col=None)
        else:
            print("You can pass only csv or xlsx")
        df = df.dropna(axis='columns', how ='all')
        df = df.dropna(axis = 0, how = 'all')
        df = df.head(1000)
        df = df.iloc[:,9:]
        return df
    except:
        print("Bad format")
        return None
    
def plot_timeseries(df):
    df.plot(subplots=True,figsize=(400,400))
    plt.show()

def stationarity_check(df,signif=0.05):
    non_stat_cols=[]
    for column in df.columns:
        ts = df[column]
        dftest = adfuller(ts, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
        for key,value in dftest[4].items():
            adf['Critical Value (%s)'%key] = value
        p = adf['p-value']
        if p <= signif:
            stationarity = 1
        else:
            stationarity = 0
            non_stat_cols.append(column)
    return non_stat_cols
    
def make_stationary(df):
    df_copy = df.copy()
    non_stat_cols= stationarity_check(df_copy)
    column_diff_dict = dict.fromkeys(non_stat_cols, 0)
    if len(non_stat_cols) != 0:
        for column in non_stat_cols:
            f = FracdiffStat()
            df_copy[column]= pd.Series(f.fit_transform(df_copy[column].values.reshape(-1,1)).squeeze())
            column_diff_dict[column] = (f.d_[0],f.window)
    else:
        return df_copy,column_diff_dict
    return df_copy,column_diff_dict

def plot_correlogram(x, fig, axes,lags=40, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x=x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    
def plot_pred_graphs(ts,predict_ts,val_ts,widget_method,title,fig, ax1, ax2):
    if widget_method == "BoDT" or widget_method == "LSTM":
        ts_copy = ts[:-len(val_ts)]
        ts_copy = ts_copy.append(val_ts,ignore_index=True)
        val_ts = ts_copy.tail(len(val_ts))
        ts_copy = ts.append(predict_ts,ignore_index=True)
        predict_ts = ts_copy.tail(len(predict_ts))
        del ts_copy
    ax1.plot(ts,'blue',label='Actual')
    ax1.plot(val_ts,'green',label='Validation')
    ax2.plot(ts,'blue',label='Actual')
    ax2.plot(predict_ts,'green',label='Predicted')
    ax1.set_title("Validation")
    ax2.set_title("Nstep prediction")
    ax1.legend()
    ax2.legend()
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

def hurst(ts):
    lags = range(2, 100)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(log(lags), log(tau), 1)
    return poly[0]*2.0


def grangers_causation_matrix(data, variables, maxlag=40,test='ssr_chi2test', verbose=False): 
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def find_cointegrated_pairs(dataframe, critial_level = 0.05):
    n = dataframe.shape[1] 
    pvalue_matrix = np.ones((n, n)) 
    keys = dataframe.columns 
    pairs = [] 
    for i in range(n):
        for j in range(i+1, n): 
            series1 = dataframe[keys[i]] 
            series2 = dataframe[keys[j]]
            result = sm.tsa.stattools.coint(series1, series2) 
            pvalue = result[1] 
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level: 
                pairs.append((keys[i], keys[j], pvalue)) 
    integrated_columns = [] 
    for pair in pairs:
        integrated_columns.append(pair[0])
        integrated_columns.append(pair[1])
    integrated_columns=list(set(integrated_columns))
    return integrated_columns, pairs

def param_calc(arr,alpha,nlags,nobs):
    var = np.ones(nlags + 1) / nobs
    var[0] = 0
    var[1] = 1. / nobs
    var[2:] *= 1 + 2 * np.cumsum(arr[1:-1]**2)
    interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(var)
    count = 0
    max_count = 0
    for i in range(1,len(interval)):
        if arr[i]>interval[i] or arr[i]< - interval[i]:
            count+=1
        else:
            count = 0
        if count>max_count:
            max_count = count
    return max_count

def pq_calc(ts):
    alpha = 0.05
    nobs  = len(ts)
    nlags = min(40,len(ts)//2 - 1)
    acf, _ = sm.tsa.acf(ts,nlags = nlags, alpha=alpha)
    pacf, _ = sm.tsa.pacf(ts,nlags = nlags, alpha=alpha)
    p = param_calc(pacf,alpha,nlags,nobs)
    q = param_calc(acf,alpha,nlags,nobs)
    return p,q
    
def get_single_columns(df,integrated_columns):
    single_columns = list(set(df.columns) - set(integrated_columns))
    return single_columns

def common_counter(coeff_list):
    most_common_list =Counter(coeff_list).most_common()
    high = most_common_list[0][1]
    h_index= most_common_list[0][0]
    for pair in most_common_list:
        if pair[1] == high:
            h_index = pair[0]
        else:
            continue
    return h_index

def multi_single_ts(df):
    integrated_columns, pairs = find_cointegrated_pairs(df)
    single_columns = list(set(df.columns) - set(integrated_columns))
    return df[integrated_columns], df[single_columns]

def invert_diff(df_forecast, columns_diff_dict):
    df_fc = df_forecast.copy()
    columns = columns_diff_dict.keys()
    for col in columns:
        f = Fracdiff(d = - columns_diff_dict[col][0],window = columns_diff_dict[col][1])
        diff = f.fit_transform(df_forecast[col].values.reshape(-1,1))
        df_fc[col] = pd.Series(diff.squeeze())
    return df_fc

def series_to_supervised(data, n_in=30, dropnan=True, predict = True):
    c_names = data.columns
    n_out=2
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += ['%s(t-%d)' % (n, i) for n in c_names]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % n) for n in c_names]
        else:
            names += [('%s(t+%d)' % (n, i)) for n in c_names]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    test = agg.tail(1)
    if dropnan:
        agg.dropna(inplace=True)
    if predict:
        agg = agg.append(test,ignore_index=True)
    return agg
