import pandas as pd
from scipy import stats
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy.linalg import cholesky, solve_triangular, cho_solve, cho_factor
from scipy.linalg import solve
from scipy.optimize import minimize
import random

# mount data to data frame
df_bbca = pd.read_csv('BBCA.JK.csv')
df_itmg = pd.read_csv('ITMG.JK.csv')
df_mcas = pd.read_csv('MCAS.JK.csv')

# Filter the column that is used
df_bbca = df_bbca[['Date', 'Symbol', 'Close','Return+1']]
df_itmg = df_itmg[['Date', 'Symbol', 'Close','Return+1']]
df_mcas = df_mcas[['Date', 'Symbol', 'Close','Return+1']]

# calculate Geometric Mean of Return
def geomeanReturn(df):
    geomean = (stats.gmean(df['Return+1'].iloc[2:-1])) - 1
    return geomean

bbca_geomean = geomeanReturn(df_bbca)
itmg_geomean = geomeanReturn(df_itmg)
mcas_geomean = geomeanReturn(df_mcas)

# calculate variance of Return as Risk
def varianceRisk(df):
    variance = df['Return+1'].iloc[2:-1].var()
    return variance

bbca_var = varianceRisk(df_bbca)
itmg_var = varianceRisk(df_itmg)
mcas_var = varianceRisk(df_mcas)

# calculate covariance of Return for each pair of stocks
def covarStock(df1,df2):
    covar = np.cov(df1['Return+1'], df2['Return+1'])
    return covar[0][1]

cov_bbca_itmg = covarStock(df_bbca,df_itmg)
cov_bbca_mcas = covarStock(df_bbca,df_mcas)
cov_itmg_mcas = covarStock(df_itmg,df_mcas)

# Here we create a random distribution of stocks to populate the scatter plot
def generateDist(n):
    arrN = [None] * n 
    arrX, arrY, arrZ = []  , [] , []
    for i in range(0,n):
        arrN[i] = random.randrange(100)

        x = arrN[i]
        arrX.append((x/100))

        y = random.randrange(100-arrN[i])
        arrY.append((y/100))

        z = 100 - x - y
        arrZ.append((z/100))
    df = pd.DataFrame(
    {'asset_1': arrX,
     'asset_2': arrY,
     'asset_3': arrZ
    })
    return df
df_dist = generateDist(100)

df_dist['a1square'] = df_dist.asset_1 ** 2
df_dist['a2square'] = df_dist.asset_2 ** 2
df_dist['a3square'] = df_dist.asset_3 ** 2

df_dist["risk"] = (df_dist.a1square * bbca_var) + (df_dist.a2square * itmg_var) + (df_dist.a3square * mcas_var) + (2*(df_dist.asset_1 * df_dist.asset_2 * cov_bbca_itmg)) + (2*(df_dist.asset_1 * df_dist.asset_3 * cov_bbca_mcas)) + (2*(df_dist.asset_3 * df_dist.asset_2 * cov_itmg_mcas))

df_dist["return"] = (df_dist.asset_1 * bbca_geomean) + (df_dist.asset_2 * itmg_geomean) + (df_dist.asset_3 * mcas_geomean)

