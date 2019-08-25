import pandas as pd
import numpy as np
import setting
from lightgbm import LGBMRegressor
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
data.drop(columns=[
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
], inplace=True)
test = data.iloc[4658147:, :]
data = data.iloc[:4658147, :]

train = pd.read_csv(f"{DATA}/train.csv", dtype={
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}, usecols=['type', 'scalar_coupling_constant'])
y = train.scalar_coupling_constant

model = {}

for bond in pd.unique(train['type']):
    X = data[train['type'] == bond]
    Y = y[train['type'] == bond]
    lgbm = LGBMRegressor(objective='regression_l1', n_estimators=5000, learning_rate=0.1, subsample_freq=1, \
                     feature_fraction=0.7, subsample=0.7, reg_alpha=0.1, reg_lambda=0.3, device_type='gpu',
                    **setting.param[bond])
    lgbm.fit(X, Y, eval_metric='regression_l1', verbose=100)
    printf(f"Saving model as {bond}.5000.lightgbm")
    lgbm.booster_.save_model(f"{bond}.5000.lightgbm")
    model[bond] = lgbm

test_ = pd.read_csv(f"{DATA}/test.csv", dtype={
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category'
}, usecols=['type'])
test = test.reset_index(drop=True)

submission = pd.read_csv(f"{DATA}/sample_submission.csv")
for bond in pd.unique(train['type']):
    X = test[test_['type'] == bond]
    Y = model[bond].predict(X)
    submission.loc[test_['type'] == bond, 'scalar_coupling_constant'] = Y
submission.to_csv("submission_08_25_01.csv", index=False)