from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima_model import ARMA
from collections import Counter
from itertools import chain
from lightgbm import LGBMRegressor
from data import find_cointegrated_pairs,pq_calc,common_counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np
from validation import evaluate_metric,get_best_params,cross_val_prediction
from config import hyper_space_lgbm
from validation import series_to_supervised, train_valid_split
warnings.simplefilter('ignore')

def varma_prediction(train,test,steps):
    p,q = get_var_pq_params(train)
    model = VARMAX(train,order=(p, q))
    model_fit = model.fit(disp=False)
    if not steps:
        prediction = model_fit.forecast(steps=len(test))
    else: 
        prediction = model_fit.forecast(steps=steps)
    multi_predicts_df = pd.DataFrame(prediction, columns = train.columns)
    return multi_predicts_df

def arma_prediction(train,test,steps=None):
    prediction_df = pd.DataFrame(columns = train.columns)
    for col in train.columns:
        p,q = get_arma_pq_params(train[col])
        model = ARMA(train[col], order= (p,q))
        model_fit = model.fit(disp=False)
        if not steps:
            prediction = model_fit.forecast(steps=len(test[col]))
        else: 
            prediction = model_fit.forecast(steps=steps)
        prediction_df[col] = prediction
    return prediction_df

def boosting_validation(df,train_columns,valid_size,inverting_times):
    ma = get_best_ma(df,train_columns,valid_size)
    t1_columns = [column+str("(t+1)") for column in train_columns]
    t_columns = [column+str("(t)") for column in train_columns]
    train = residuals_add(df,t_columns,ma,inverting_times,predict= False)
    train, test = train_valid_split(train,valid_size)
    prediction_df = pd.DataFrame()
    true_y = pd.DataFrame()
    for column in t1_columns:
        train_y = train[column]
        train_x = train.drop(t1_columns,axis = 1)
        test_y = test[column]
        test_x = test.drop(t1_columns,axis = 1)
        lgbm_model = LGBMRegressor()
        lgbm_best_params = get_best_params(train_x,train_y,4,lgbm_model,hyper_space_lgbm)
        lgbm_predicts = cross_val_prediction(lgbm_model,lgbm_best_params,train_x,train_y,test_x)
        column = column[:-5]
        prediction_df[column] = lgbm_predicts
        true_y[column] = test_y
    return prediction_df,true_y,ma,lgbm_best_params

def boosting_prediction(df,train_columns,steps,ma,best_params,inverting_times):
    t1_columns = [column+str("(t+1)") for column in train_columns]
    t_columns = [column+str("(t)") for column in train_columns]
    train = residuals_add(df,t_columns,ma,inverting_times,predict= True)
    test = train.tail(1)
    train = train.dropna(axis = 0)
    prediction_df = pd.DataFrame()
    all_predicts = pd.DataFrame(columns = train_columns)
    for column in t1_columns:
        train_y = train[column]
        train_x = train.drop(t1_columns,axis = 1)
        test_y = test[column]
        test_x = test.drop(t1_columns,axis = 1)
        lgbm_model = LGBMRegressor()
        lgbm_predicts = cross_val_prediction(lgbm_model,best_params,train_x,train_y,test_x)
        column = column[:-5]
        prediction_df[column] = lgbm_predicts
    df = df.append(prediction_df,ignore_index=True)
    all_predicts = all_predicts.append(prediction_df,ignore_index=True)
    for i in range(steps-1):
        train = residuals_add(df,t_columns,ma,inverting_times,predict= True)
        test = train.tail(1)
        train = train.dropna(axis = 0)
        prediction_df = pd.DataFrame()
        for column in t1_columns:
            train_y = train[column]
            train_x = train.drop(t1_columns,axis = 1)
            test_y = test[column]
            test_x = test.drop(t1_columns,axis = 1)
            lgbm_model = LGBMRegressor()
            lgbm_predicts = cross_val_prediction(lgbm_model,best_params,train_x,train_y,test_x)
            column = column[:-5]
            prediction_df[column] = lgbm_predicts
        df = df.append(prediction_df,ignore_index=True)
        all_predicts = all_predicts.append(prediction_df,ignore_index=True)
    return all_predicts

def get_best_ma(df,train_columns,valid_size):
    t1_columns = [column+str("(t+1)") for column in train_columns]
    t_columns = [column+str("(t)") for column in train_columns]
    best_rmse = np.inf
    best_ma = 0
    break_count = 0
    mas = np.arange(10,df.shape[0]//2,10)
    for ma in mas:
        train = series_to_supervised(df,ma,predict= False)
        train, test = train_valid_split(train,valid_size)
        prediction_df = pd.DataFrame()
        kf = TimeSeriesSplit(n_splits=10)
        true_y = pd.DataFrame()
        for column in t1_columns[:1]:
            train_y = train[column]
            train_x = train.drop(t1_columns,axis = 1)
            test_y = test[column]
            test_x = test.drop(t1_columns,axis = 1)
            model = LGBMRegressor()
            pred = cross_val_prediction(model,None,train_x,train_y,test_x)
            mse = mean_squared_error(test_y, pred)
            if mse < best_rmse:
                best_rmse = mse
                best_ma = ma
                break_count = 0
            else:
                break_count+=1
            if break_count==3:
                break
    return best_ma

def get_var_pq_params(integrated_ts):
    p_coeffs = []
    q_coeffs = []
    for col in integrated_ts.columns:
        p,q = pq_calc(integrated_ts[col])
        p_coeffs.append(p)
        q_coeffs.append(q)
    p = common_counter(p_coeffs)
    q = common_counter(q_coeffs)
    if p == 0 and q == 0:
        p = 1
    return p,q

def residuals_add(train,train_columns,ma,inverting_times,predict):
    train = series_to_supervised(train,n_in = ma,predict = predict)
    resid_columns = ['ARIMA_'+str(column) for column in train_columns] + ['HWES_'+str(column) for column in train_columns]
    resid_df = pd.DataFrame(columns = resid_columns)
    for column in train_columns:
        p,q = pq_calc(train[column])
        i = 1
        if column in inverting_times.keys():
            d = 1
        else:
            d = 0
        model = ARIMA(train[column], order= (p,d,q))
        while True:
            try:
                model_fit = model.fit()
            except:
                i+=1
                p,q = p-1, q-1
                model = ARIMA(train[column], order= (p,d,q))
            else:
                break
        resid_df['ARIMA_'+str(column)] = model_fit.resid
        if resid_df['ARIMA_'+str(column)].isna().any():
            resid_df = resid_df.drop('ARIMA_'+str(column), axis = 1)
        model = ExponentialSmoothing(train[column])
        model_fit = model.fit()
        resid_df['HWES_'+str(column)] = model_fit.resid
        if resid_df['HWES_'+str(column)].isna().any():
            resid_df = resid_df.drop('HWES_'+str(column), axis = 1)
    train = pd.concat([train,resid_df],axis = 1)
    return train