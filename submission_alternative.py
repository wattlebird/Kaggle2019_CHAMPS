import pandas as pd
import numpy as np
import setting
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
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
tdata = data.iloc[4658147:, :]
tdata = tdata.reset_index(drop=True)
data = data.iloc[:4658147, :]

train = pd.read_csv(f"{DATA}/train.csv", dtype={
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}, usecols=['molecule_name', 'type', 'scalar_coupling_constant'])
test = pd.read_csv(f"{DATA}/test.csv", dtype={
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category'
}, usecols=['type'])
y = train.scalar_coupling_constant
submission = pd.read_csv(f"{DATA}/sample_submission.csv")

for bond in pd.unique(train['type']):
    X = data[train['type'] == bond]
    Y = y[train['type'] == bond]
    y_pred = np.zeros(tdata[test['type'] == bond].shape[0], dtype='float32')
    gkf = GroupKFold(2)
    for i, (it, iv) in enumerate(gkf.split(X, Y, groups=train[train['type']==bond].molecule_name)):
        lgbm = LGBMRegressor(objective='regression_l1', n_estimators=10000, learning_rate=0.1, subsample_freq=1, \
                         feature_fraction=0.7, subsample=0.7, reg_alpha=0.1, reg_lambda=0.3, device_type='gpu',
                        **setting.param[bond])
        lgbm.fit(
            X.iloc[it],
            Y.iloc[it],
            eval_set=[(X.iloc[it], Y.iloc[it]), (X.iloc[iv], Y.iloc[iv])],
            eval_metric='regression_l1',
            verbose=100,
            early_stopping_rounds=200
        )
        print(f"In fold {i}, a model training stopped at iteration {lgbm.best_iteration_} with score {lgbm.best_score_['valid_1']}. Dumping model as {bond}_fold{i}.lightgbm")
        lgbm.booster_.save_model(f"{bond}_fold{i}.lightgbm")
        y_pred += lgbm.predict(tdata[test['type'] == bond])
    y_pred /= 2
    submission.loc[test['type'] == bond, 'scalar_coupling_constant'] = y_pred

submission.to_csv("submission_08_25_02.csv", index=False)