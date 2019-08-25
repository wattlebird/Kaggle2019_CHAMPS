import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import setting
import gc

gc.enable()

DATA = '~/Data/Molecular'

data = pd.concat([
    pd.read_pickle(f"{DATA}/basic.gz"),
    pd.read_pickle(f"{DATA}/angle_feature.gz"),
    pd.read_pickle(f"{DATA}/criskiev_distance_feature.gz"),
    pd.read_pickle(f"{DATA}/qm9.gz")
], axis = 1)
data['atom_2'] = data['atom_2'].astype('category')
data['atom_3'] = data['atom_3'].astype('category')
data['atom_4'] = data['atom_4'].astype('category')
data['atom_5'] = data['atom_5'].astype('category')
data['atom_6'] = data['atom_6'].astype('category')
data['atom_7'] = data['atom_7'].astype('category')
data['atom_8'] = data['atom_8'].astype('category')
data['atom_9'] = data['atom_9'].astype('category')
data = data.iloc[:4658147, :]
data = data.drop(columns=[
    'id',
    'molecule_name',
    'atom_index_0',
    'atom_index_1',
    'type',
    'atom_0',
    'x_0',
    'y_0',
    'z_0',
    'atom_1',
    'x_1',
    'y_1',
    'z_1'
])

train = pd.read_csv(f"{DATA}/train.csv", dtype={
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}, usecols=['type', 'scalar_coupling_constant'])
y = train.scalar_coupling_constant

def foo(typ, subsample, bagging_fraction):
    X, Xv, yt, yv = train_test_split(data[train['type']==typ], y[train['type']==typ], test_size=0.3, random_state=233)
    lgbm = LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.1, 
        subsample_freq=1, 
        subsample=subsample, 
        bagging_fraction=bagging_fraction, 
        reg_alpha=0.1, 
        reg_lambda=0.3,
        device_type='gpu',
        **setting.param[typ]
    )
    lgbm.fit(X, yt, eval_metric='regression_l1', verbose=100)
    return -1*mean_absolute_error(yv, lgbm.predict(Xv))

for bond in pd.unique(train['type']):
    optimizer = BayesianOptimization(
        f=partial(foo, typ=bond),
        pbounds={
            'subsample': (0.1, 0.99),
            'bagging_fraction': (0.1, 0.99)
        },
        random_state=233,
        verbose=2
    )
    logger = JSONLogger(path=f"./{bond}-bayesian-optimization.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.maximize(acq="ei", xi=1e-3)