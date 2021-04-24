from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

hyper_space_lgbm = {
    'learning_rate':    hp.uniform('learning_rate',0.001,0.1),
    'num_leaves':       hp.quniform('num_leaves', 31, 100, q=1),
    'min_split_gain':   hp.uniform('min_split_gain', 0, 1),
    'reg_alpha':        hp.uniform('reg_alpha',0,1),
    'reg_lambda':       hp.uniform('reg_lambda',0,1),
    'n_estimators':     hp.quniform('n_estimators', 100, 250, q=1)
}