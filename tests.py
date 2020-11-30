%load_ext autoreload
%autoreload 2

import pandas as pd
from sklearn import preprocessing
from linear_models import stepwise_linear_ols


# ====== TEST LINEAR_MODELS ======

# --- LOAD DATA ---

df = pd.read_csv(r'/Users/howard/Dropbox/02_WORK/05_Projects/01_Research/05_BoardEx/01-PREP/dat/05-MODEL_DATA_MNTH-v02-20201129.csv', dtype={'CID': str, 'SIC': str})

# DROPNA
df2 = df.dropna(subset=['EVT_CHGS_IN', 'EVT_CHGS_OUT', 'EFF_CHGS_IN', 'EFF_CHGS_OUT', 'ANN_CHGS_IN', 'ANN_CHGS_OUT'], how='all')
df2 = df2.dropna(subset=['RESIDUAL_M0', 'RESIDUAL_M1'], how='any')

# FILLNA 0
cols_na = ['MEMBERS', 'EVT_CHGS_IN', 'EVT_CHGS_OUT', 'EFF_CHGS_IN', 'EFF_CHGS_OUT', 'ANN_CHGS_IN', 'ANN_CHGS_OUT']
df2[cols_na] = df2[cols_na].fillna(0)

# Fillna MEAN
cols_avg = ['ME', 'PRICE', 'EQTVAL', 'EQTVOL']
df2[cols_avg] = df2[cols_avg].fillna(df2[cols_avg].mean())

# SIC
df2['SIC1'] = df2['SIC'].str[0]

# ADD YEAR
df2['YEAR'] = df2['DATE'].str[0:4]

# DUMMY FOR SIC
dm_sic = pd.get_dummies(df2['SIC1'])
dm_sic.columns = ['SIC_'+str(k) for k in dm_sic.columns]

# DUMMY FOR YEAR
dm_year = pd.get_dummies(df2['YEAR'])
dm_year.columns = ['YEAR_'+str(k) for k in dm_year.columns]

# MERGE DUMMIES
cols2 = ['RETADJ', 'ME', 'SHROUT', 'PRICE', 'MEMBERS',
        'EVT_CHGS_IN', 'EVT_CHGS_OUT', 'EFF_CHGS_IN', 'EFF_CHGS_OUT', 'ANN_CHGS_IN', 'ANN_CHGS_OUT', 'RESIDUAL_M0', 'RESIDUAL_M1',
        'MEMBERS2', 'EQTVAL', 'EQTVOL', 'IS_MALE'] + ['EDU_V{:03d}'.format(int(k)+1) for k in range(16)] + ['EXP_V{:03d}'.format(int(k)+1) for k in range(16)]
df3 = pd.concat([df2[cols2], dm_sic, dm_year], axis=1)

# STANDARDIZE
df3['ME'] = preprocessing.scale(df3['ME'])
df3['SHROUT'] = preprocessing.scale(df3['SHROUT'])
df3['PRICE'] = preprocessing.scale(df3['PRICE'])
df3['EQTVAL'] = preprocessing.scale(df3['EQTVAL'])
df3['EQTVOL'] = preprocessing.scale(df3['EQTVOL'])



# --- MODEL TRAINING ---

cols3 = ['RETADJ', 'ME', 'SHROUT', 'PRICE', 'MEMBERS',
         'EVT_CHGS_IN', 'EVT_CHGS_OUT', 'EFF_CHGS_IN', 'EFF_CHGS_OUT', 'ANN_CHGS_IN', 'ANN_CHGS_OUT',
         'MEMBERS2', 'IS_MALE'] + ['EDU_V{:03d}'.format(int(k)+1) for k in range(16)] + ['EXP_V{:03d}'.format(int(k)+1) for k in range(16)] +\
        dm_sic.columns.tolist() + dm_year.columns.tolist()

res = stepwise_linear_ols(df3, depvar='RESIDUAL_M0', indepvars=cols3, maxstep=100)