import os
import sys
import pandas as pd


# SETUP
lna = len(sys.argv)

if (lna-2)%3 != 0:
    print('Need to have #args 4, 7, 10...')

else:
    npr = int((lna-2)/3)

    for n in range(npr):
        bs = n * 3 + 1

        # FILE
        f1 = sys.argv[bs]
        f2 = sys.argv[bs+3]

        # FILENAME
        fn1 = os.path.basename(f1)
        fn2 = os.path.basename(f2)

        # COLUMNS
        lsl = sys.argv[bs+1].split(',')
        if n < npr-1:
            lsr = list(set(sys.argv[bs+2].split(',') + sys.argv[bs+4].split(',')))
        else:
            lsr = sys.argv[bs+2].split(',')

        # LOAD DATA
        dfl = pd.read_csv(f1, usecols=lsl, dtype=dict((k,'str') for k in lsl))
        dfr = pd.read_csv(f2, usecols=lsr, dtype=dict((k,'str') for k in lsr))

        # MERGE
        dfl_dp = dfl[lsl].drop_duplicates()
        dfr_dp = dfr[lsr].drop_duplicates()
        mg = dfl_dp.merge(dfr_dp, left_on=lsl, right_on=lsr)

        print('{0} | {1:,d} = {2:,d} - {3:,d} - {4:,d} = {5:,d} | {6}'.format(fn1, dfl.shape[0], dfl_dp.shape[0], mg.shape[0], dfr_dp.shape[0], dfr.shape[0], fn2))
