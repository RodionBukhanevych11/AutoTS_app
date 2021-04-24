from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import warnings
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import partial
import numpy as np

warnings.simplefilter('ignore')

def train_valid_split(df,size=0.8):
    train = df[:int(size*(len(df)))]
    valid = df[int(size*(len(df))):]
    return train,valid

def train_test_split_func(X,y,size=0.25):
    train_x, val_x, train_y, val_y = train_test_split(X,y,test_size=size)
    return train_x, val_x, train_y, val_y

def metrics(df_true,df_pred):
    val_metrics = pd.DataFrame(columns=['column','mae','mse'])
    for i,column in enumerate(df_true):
        mae=mean_absolute_error(df_true[column],df_pred[column])
        mse=mean_squared_error(df_true[column],df_pred[column])
        val_metrics = val_metrics.append({'column':column,'mae':mae,'mse':mse},ignore_index=True)
    return val_metrics

def evaluate_metric(params,X_train,Y_train,max_evals,model):
    if params is not None:
        if type(model) is lightgbm.sklearn.LGBMRegressor:
            params['n_estimators'] = np.int(params['n_estimators'])
            params['num_leaves'] = np.int(params['num_leaves'])
        model.set_params(**params)
    model.set_params(**params)
    num_splits = 5
    rmse_score = 0
    kf = TimeSeriesSplit(n_splits=num_splits)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        y_train, y_valid = pd.DataFrame(Y_train.iloc[train_index].copy()), pd.DataFrame(Y_train.iloc[test_index])
        x_train, x_valid = pd.DataFrame(X_train.iloc[train_index, :].copy()), pd.DataFrame(X_train.iloc[test_index, :].copy())
        fit_model = model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        rmse_score += mean_squared_error(y_valid, pred)
        del y_valid, x_train, x_valid, y_train
    rmse_score/=num_splits
    return  {'loss': rmse_score, 'params': params,'status': STATUS_OK}

def get_best_params(X_train,Y_train,max_evals,model,hyper_space):
    trials = Trials()
    best_params = fmin( 
                fn=partial(evaluate_metric, X_train=X_train, Y_train=Y_train,max_evals=max_evals,model = model),
                space=hyper_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.RandomState(1),
                show_progressbar=True
            )
    return best_params

def series_to_supervised(data, n_in=30,n_out= 2, predict=True):
    c_names = data.columns
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
    agg.dropna(inplace=True)
    if predict:
        agg = agg.append(test,ignore_index=True)
    return agg

def cross_val_prediction(model,params,X_train,Y_train,X_test):
    if params is not None:
        if type(model) is lightgbm.sklearn.LGBMRegressor:
            params['n_estimators'] = np.int(params['n_estimators'])
            params['num_leaves'] = np.int(params['num_leaves'])
        model.set_params(**params)
    num_splits = 5
    kf = TimeSeriesSplit(n_splits=num_splits)
    y_test_pred = pd.Series([0 for i in range(len(X_test))])
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        y_train, y_valid = pd.DataFrame(Y_train.iloc[train_index].copy()), pd.DataFrame(Y_train.iloc[test_index])
        x_train, x_valid = pd.DataFrame(X_train.iloc[train_index, :].copy()), pd.DataFrame(X_train.iloc[test_index, :].copy())
        fit_model = model.fit(x_train, y_train)
        pred = model.predict(X_test)
        y_test_pred += pred.squeeze()
        del y_valid, x_train, x_valid, y_train
    y_test_pred = y_test_pred / num_splits
    return y_test_pred