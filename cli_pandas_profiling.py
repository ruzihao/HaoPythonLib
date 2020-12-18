import sys
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


# SETUP
lna = len(sys.argv)

if sys.argv[1] == '-m':
    d = 2
else:
    d = 1


# FILES
ifl = sys.argv[d]
ofl = sys.argv[d+1]

# CONVERT TO STR
lsr = []
if len(sys.argv)>d+2:
    lsr = sys.argv[d+2].split(',')

# LOAD DATA
df = pd.read_csv(ifl, dtype=dict((k, 'str') for k in lsr))

# CONVERT TO LOG
llg = []
if len(sys.argv)>d+3:
    llg = sys.argv[d+3].split(',')

for k in llg:
    df['[LOG]{0}'.format(k)] = np.log(df[k])
    
# ANALYSIS
if sys.argv[1] == '-m':
    prof = ProfileReport(df, pool_size=4, minimal=True)
else:
    prof = ProfileReport(df, pool_size=4)

# EXPORT
prof.to_file(ofl)

