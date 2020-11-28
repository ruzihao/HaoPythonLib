import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calc_vif(df_ori, indep_vars=None):
    indep_vars = indep_vars or df_ori.columns
    df = add_constant(df_ori[indep_vars])
    dct = {}
    for idx, col in enumerate(indep_vars):
        dct[col] = variance_inflation_factor(df.values, idx)
    return pd.Series(dct)


def calc_woe_iv(desc, df_ori, label, indep_vars=None, nbins=10):
    indep_vars = indep_vars + [label] if indep_vars is not None else list(df_ori.columns)
    df = df_ori[indep_vars]
    for feat in [k for k in indep_vars if k != label]:
        tmp = df[[feat, label]]
        if str(tmp[feat].dtype) != 'object' and (tmp[feat].nunique() >= nbins):
            tmp.loc[:, 'group'] = pd.qcut(tmp[feat], q=nbins, labels=False, duplicates='drop')
            tmp.loc[:, 'group'] = 0 if tmp['group'].count()==0 else tmp['group']
        else:
            tmp.loc[:, 'group'] = tmp[feat]
        tmp_sum = tmp.assign(dmy=1).groupby('group').agg({'dmy': 'count', label: ('sum', 'mean')})
        tmp_sum.columns = ['count', 'hk_count', 'avg_is_hk']
        tmp_sum = tmp_sum[['count', 'hk_count', 'avg_is_hk']].reset_index()
        all_bad = (tmp[label]==1).sum()
        all_good = (tmp[label]==0).sum()
        all_ratio = all_good/all_bad
        if not tmp_sum.empty:
            for i in np.arange(tmp_sum.shape[0]):
                good = tmp_sum.loc[tmp_sum.index[i], 'count'] - tmp_sum.loc[tmp_sum.index[i], 'hk_count']
                bad = tmp_sum.loc[tmp_sum.index[i], 'hk_count']
                tmp_sum.loc[tmp_sum.index[i], 'woe'] = np.log(all_ratio*((1+bad)/(good+1)))
                tmp_sum.loc[tmp_sum.index[i], 'iv'] = (bad/all_bad-good/all_good)*tmp_sum.loc[tmp_sum.index[i], 'woe']
            desc.loc[feat, 'IV'] = tmp_sum['iv'].sum()
        else:
            desc.loc[feat, 'IV'] = 0
    return desc


def EDD(data, dd, dep=None, out=None, corr=None, iv=False, verbose=False):
    desc_trans = data.describe(include='all', percentiles=[.01, .05, .25, .5, .75, .95, .99]).transpose()
    dty = data.dtypes
    dty.name = 'dtype'
    desc_conc = pd.concat([dty, desc_trans], axis=1)

    # columns to keep
    cols_to_keep = ['type', 'dtype', 'count', 'unique', 'top', 'freq', '1%', '5%', '25%', '50%', '75%', '95%', '99%',
                    'max']
    desc_conc.loc[:, 'top'] = desc_conc.get('top', pd.Series(np.nan, index=desc_conc.index)).combine_first(
        desc_conc.get('mean', pd.Series(np.nan, index=desc_conc.index)))
    desc_conc.loc[:, 'count'] = data.shape[0] - desc_conc['count']
    desc_conc.loc[:, 'freq'] = desc_conc.get('freq', pd.Series(np.nan, index=desc_conc.index)).combine_first(
        desc_conc.get('min', pd.Series(np.nan, index=desc_conc.index)))
    desc_conc.loc[:, 'type'] = desc_conc['dtype'].apply(
        lambda x: 'CHAR' if x == 'object' else ('DATETIME' if 'datetime' in str(x) else 'NUM'))
    desc_conc = desc_conc[[k for k in cols_to_keep if k in desc_conc.columns]]

    # rename
    desc_conc = desc_conc.rename(columns={'count': 'num_missing',
                                          'unique': 'num_unique',
                                          'top': 'mean_or_top1',
                                          'freq': 'min_or_top2',
                                          '1%': 'p1_or_top3',
                                          '5%': 'p5_or_top4',
                                          '25%': 'p25_or_top5',
                                          '50%': 'p50_or_bottom5',
                                          '75%': 'p75_or_bottom4',
                                          '95%': 'p95_or_bottom3',
                                          '99%': 'p99_or_bottom2',
                                          'max': 'max_or_bottom1'})

    cols_corr = []
    # handle num_unique in numeric columns
    print('Numeric...')
    for var in desc_conc[desc_conc['type'] == 'NUM'].index:
        print(var)
        desc_conc.loc[var, 'num_unique'] = data[var].nunique()
        desc_conc.loc[var, 'std'] = data[var].dropna().std()
        desc_conc.loc[var, 'range'] = desc_conc.loc[var, 'max_or_bottom1'] - desc_conc.loc[var, 'min_or_top2']
        desc_conc.loc[var, 'IQR'] = desc_conc.loc[var, 'p75_or_bottom4'] - desc_conc.loc[var, 'p25_or_top5']
        desc_conc.loc[var, 'skewness'] = skew(data[var].dropna())
        desc_conc.loc[var, 'excess_kurtosis'] = kurtosis(data[var].dropna())
        if dep is not None:
            desc_conc.loc[var, 'pearson_corr'] = pearsonr(x=data[var], y=data[dep].astype(float))[0]
            if data[var].nunique() > 1:
                desc_conc.loc[var, 'spearman_corr'] = spearmanr(a=data[var], b=data[dep].astype(float))[0]
            cols_corr = ['pearson_corr', 'spearman_corr']
        if not str(desc_conc.loc[var, 'dtype']).startswith('datetime'):
            desc_conc.loc[var, 'num_zeros'] = sum(data[var] == 0)
        desc_conc.loc[var, 'pct_zeros'] = desc_conc.loc[var, 'num_zeros'] / data.shape[0]

    if corr is not None:
        # df_X = dat[list(desc_conc[desc_conc['type']=='NUM'].index)]
        df_corr = data[corr].astype(float).corr()

    # handle "object" columns
    print('Categorical...')
    dct_val_cnts = {}
    cols_top = ['mean_or_top1', 'min_or_top2', 'p1_or_top3', 'p5_or_top4', 'p25_or_top5']
    cols_bot = ['p50_or_bottom5', 'p75_or_bottom4', 'p95_or_bottom3', 'p99_or_bottom2', 'max_or_bottom1']
    for var in desc_conc[desc_conc['type'] == 'CHAR'].index:
        print(var)
        val_cnts = data[var].where(data[var] != 'nan', None).value_counts()
        tot_cnt = val_cnts.sum()
        if verbose and dd.loc[dd['variable'] == var, 'verbose'].iloc[0] == 1 and val_cnts.shape[0] > 1:
            dct_val_cnts[var] = val_cnts.reset_index().reset_index()
            dct_val_cnts[var].columns = ['index', 'value', 'freq']
            dct_val_cnts[var].loc[:, 'index'] += 1
        top5 = val_cnts.head(5)
        if val_cnts.shape[0] < 5:
            bot5 = None
        elif val_cnts.shape[0] < 10:
            bot5 = val_cnts.iloc[5:]
        else:
            bot5 = val_cnts.tail(5)

        if top5 is not None:
            top5_flat = [':'.join([str(k[0]), str(k[1]), str(np.round(k[1]/tot_cnt, 4))]) for k in zip(top5.index, top5)]
            if len(top5_flat) < 5:
                top5_flat += [''] * (5 - len(top5_flat))
            for idx in range(len(top5_flat)):
                desc_conc.loc[var, cols_top[idx]] = top5_flat[idx]
        if bot5 is not None:
            bot5_flat = [':'.join([str(k[0]), str(k[1]), str(np.round(k[1]/tot_cnt, 4))]) for k in zip(bot5.index, bot5)]
            if len(bot5_flat) < 5:
                bot5_flat += [''] * (5 - len(bot5_flat))
            for idx in range(len(bot5_flat)):
                desc_conc.loc[var, cols_bot[idx]] = bot5_flat[idx]
        desc_conc.loc[var, 'num_missing'] = data.shape[0] - val_cnts.sum()
        desc_conc.loc[var, 'num_unique'] = val_cnts.shape[0]
    desc_conc.loc[:, 'nobs'] = data.shape[0]
    desc_conc.loc[:, 'pct_missing'] = desc_conc['num_missing'] / desc_conc['nobs']
    desc_conc = desc_conc[
        [k for k in ['type', 'dtype', 'nobs', 'num_missing', 'pct_missing', 'num_unique', 'num_zeros', 'pct_zeros',
                     'std', 'range', 'IQR', 'skewness', 'excess_kurtosis', 'mean_or_top1', 'min_or_top2', 'p1_or_top3',
                     'p5_or_top4', 'p25_or_top5',
                     'p50_or_bottom5', 'p75_or_bottom4', 'p95_or_bottom3', 'p99_or_bottom2',
                     'max_or_bottom1'] + cols_corr if k in desc_conc.columns]]
    desc_conc.loc[:, 'nobs'] = desc_conc['nobs'].astype(int)
    desc_conc = desc_conc.rename(columns={'index': 'Variable_Name'})
    desc_conc.loc[:, 'dtype'] = desc_conc['dtype'].astype(str)

    # handle "datetime" columns
    print('Datetime...')
    lst = []
    cols_key = ['dtype', 'nobs', 'num_missing', 'num_unique', 'num_zeros', 'type']
    for var in desc_conc[desc_conc['type'] == 'DATETIME'].index:
        print(var)
        data.loc[:, 'ym'] = data[var].apply(lambda x: str(x.year) + '{:02}'.format(x.month))
        sr = pd.concat([desc_conc.loc[var, cols_key], data['ym'].value_counts()])
        sr.name = var
        lst.append(sr)
    desc_conc = desc_conc[desc_conc['type'] != 'DATETIME']
    if len(lst):
        desc_conc2 = pd.concat(lst, axis=1).transpose()
        desc_conc2 = desc_conc2[cols_key + [k for k in desc_conc2.columns if k not in cols_key and 'nan' not in k]]
        desc_conc2.loc[:, 'dtype'] = desc_conc2['dtype'].astype(str)
    else:
        desc_conc2 = pd.DataFrame()

    if dep and iv:
        desc_conc = calc_woe_iv(desc_conc, data, dep)

    # add index.name
    desc_conc = desc_conc.reset_index().rename(columns={'index': 'variable'})
    desc_conc2 = desc_conc2.reset_index().rename(columns={'index': 'variable'})

    # add category
    desc_conc = dd.merge(desc_conc, how='right', on='variable')
    desc_conc2 = dd.merge(desc_conc2, how='right', on='variable')
    # if corr is not None:
    #     df_corr = df_cat.merge(df_corr, on='variable')

    if out is not None:
        import os
        writer = pd.ExcelWriter(os.path.join(out))
        if not desc_conc.empty:
            desc_conc.to_excel(writer, sheet_name='NUM_CHAR', index=False)
        if not desc_conc2.empty:
            desc_conc2.to_excel(writer, sheet_name='DATETIME', index=False)
        if corr is not None:
            df_corr = df_corr.reset_index().rename(columns={'index': 'variable'})
            df_corr.to_excel(writer, sheet_name='CORR', index=False)
        if verbose:
            for var in dct_val_cnts.keys():
                dct_val_cnts[var].to_excel(writer, sheet_name='_VAR_{0}'.format(var), index=False)
        writer.save()
        writer.close()
    else:
        if corr is not None:
            return {'NUM_CHAR': desc_conc, 'DATETIME': desc_conc2, 'CORR': df_corr}
        else:
            return {'NUM_CHAR': desc_conc, 'DATETIME': desc_conc2}


