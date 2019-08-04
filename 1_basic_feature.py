import pandas as pd
import numpy as np

DATA = '/home/ike/Data/Molecular/'

train = pd.read_csv(f"{DATA}train.csv")
test = pd.read_csv(f"{DATA}test.csv")
structure = pd.read_csv(f"{DATA}structures.csv")
data = pd.concat([train.drop(columns=['scalar_coupling_constant']), test], ignore_index=True)
data = data\
    .merge(structure.add_suffix("_0"), how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name_0', 'atom_index_0'])\
    .merge(structure.add_suffix("_1"), how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name_1', 'atom_index_1'])\
    .drop(columns=['molecule_name_0', 'molecule_name_1'])

data['dist_x'] = abs(data['x_0']-data['x_1'])
data['dist_y'] = abs(data['y_0']-data['y_1'])
data['dist_z'] = abs(data['z_0']-data['z_1'])
data['dist'] = np.sqrt(data['dist_x']**2 + data['dist_y']**2 + data['dist_z']**2)

data['dist_x_molecule_mean'] = data.groupby('molecule_name')['dist_x'].transform('mean')
data['dist_x_molecule_std'] = data.groupby('molecule_name')['dist_x'].transform('std')
data['dist_x_molecule_min'] = data.groupby('molecule_name')['dist_x'].transform('min')
data['dist_x_molecule_max'] = data.groupby('molecule_name')['dist_x'].transform('max')
data['dist_x_molecule_median'] = data.groupby('molecule_name')['dist_x'].transform('median')
data['dist_x_molecule_skew'] = data.groupby('molecule_name')['dist_x'].transform('skew')

data['dist_y_molecule_mean'] = data.groupby('molecule_name')['dist_y'].transform('mean')
data['dist_y_molecule_std'] = data.groupby('molecule_name')['dist_y'].transform('std')
data['dist_y_molecule_min'] = data.groupby('molecule_name')['dist_y'].transform('min')
data['dist_y_molecule_max'] = data.groupby('molecule_name')['dist_y'].transform('max')
data['dist_y_molecule_median'] = data.groupby('molecule_name')['dist_y'].transform('median')
data['dist_y_molecule_skew'] = data.groupby('molecule_name')['dist_y'].transform('skew')

data['dist_z_molecule_mean'] = data.groupby('molecule_name')['dist_z'].transform('mean')
data['dist_z_molecule_std'] = data.groupby('molecule_name')['dist_z'].transform('std')
data['dist_z_molecule_min'] = data.groupby('molecule_name')['dist_z'].transform('min')
data['dist_z_molecule_max'] = data.groupby('molecule_name')['dist_z'].transform('max')
data['dist_z_molecule_median'] = data.groupby('molecule_name')['dist_z'].transform('median')
data['dist_z_molecule_skew'] = data.groupby('molecule_name')['dist_z'].transform('skew')

data['dist_molecule_mean'] = data.groupby('molecule_name')['dist'].transform('mean')
data['dist_molecule_std'] = data.groupby('molecule_name')['dist'].transform('std')
data['dist_molecule_min'] = data.groupby('molecule_name')['dist'].transform('min')
data['dist_molecule_max'] = data.groupby('molecule_name')['dist'].transform('max')
data['dist_molecule_median'] = data.groupby('molecule_name')['dist'].transform('median')
data['dist_molecule_skew'] = data.groupby('molecule_name')['dist'].transform('skew')

data['dist_x_molecule_type_mean'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('mean')
data['dist_x_molecule_type_std'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('std')
data['dist_x_molecule_type_min'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('min')
data['dist_x_molecule_type_max'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('max')
data['dist_x_molecule_type_median'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('median')
data['dist_x_molecule_type_skew'] = data.groupby(['molecule_name', 'type'])['dist_x'].transform('skew')

data['dist_y_molecule_type_mean'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('mean')
data['dist_y_molecule_type_std'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('std')
data['dist_y_molecule_type_min'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('min')
data['dist_y_molecule_type_max'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('max')
data['dist_y_molecule_type_median'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('median')
data['dist_y_molecule_type_skew'] = data.groupby(['molecule_name', 'type'])['dist_y'].transform('skew')

data['dist_z_molecule_type_mean'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('mean')
data['dist_z_molecule_type_std'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('std')
data['dist_z_molecule_type_min'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('min')
data['dist_z_molecule_type_max'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('max')
data['dist_z_molecule_type_median'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('median')
data['dist_z_molecule_type_skew'] = data.groupby(['molecule_name', 'type'])['dist_z'].transform('skew')

data['dist_molecule_type_mean'] = data.groupby(['molecule_name', 'type'])['dist'].transform('mean')
data['dist_molecule_type_std'] = data.groupby(['molecule_name', 'type'])['dist'].transform('std')
data['dist_molecule_type_min'] = data.groupby(['molecule_name', 'type'])['dist'].transform('min')
data['dist_molecule_type_max'] = data.groupby(['molecule_name', 'type'])['dist'].transform('max')
data['dist_molecule_type_median'] = data.groupby(['molecule_name', 'type'])['dist'].transform('median')
data['dist_molecule_type_skew'] = data.groupby(['molecule_name', 'type'])['dist'].transform('skew')

data.fillna(0.0, inplace=True)

neighbour = pd.DataFrame({
    'molecule_name': np.hstack([data['molecule_name'], data['molecule_name']]),
    'atom_index_0': np.hstack([data['atom_index_0'], data['atom_index_1']]),
    'atom_index_1': np.hstack([data['atom_index_1'], data['atom_index_0']]),
    'atom_0': np.hstack([data['atom_0'], data['atom_1']]),
    'atom_1': np.hstack([data['atom_1'], data['atom_0']]),
    'type': np.hstack([data['type'], data['type']]),
    'dist_x': np.hstack([data['dist_x'], data['dist_x']]),
    'dist_y': np.hstack([data['dist_y'], data['dist_y']]),
    'dist_z': np.hstack([data['dist_z'], data['dist_z']]),
    'dist': np.hstack([data['dist'], data['dist']]),
}, columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_0', 'atom_1', 'type', 'dist_x', 'dist_y', 'dist_z', 'dist'])