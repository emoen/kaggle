import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = pd.concat([train, test], keys=['train', 'test']) 

#drop cols
features.drop(['Utilities', 'BsmtFinSFB', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', 'CSsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)

#droped - keep: RoofMatl, 'Heating', (Functional), (GarageCond), PoolQC- type('O')
#droped - keep: BsmtFullBath  - type int64 -tostring()
#droped - WoodDeckSF(.32), OpenPorchSF(.32) EnclosedPorch(-0.12), 3SsnPorch(0.04), ScreenPorch(0.11), PoolArea(0.09),  - int64
#droped: , BsmtFinSFB (-0.01), BsmtUnfSF (0.21),  - float 64
#!droped: LotFrontage(0.35), 'MasVnrArea' (corr:0.48) , 'BsmtFinSFA'(0.39) - float 64
#delete: 'Utilities', 'LowQualFinSF', BsmtHalfBath
    
dofillna()
difNames(features)

#Out - liars!
features = features.drop(features[features['Id'] == 1299].index)
features = features.drop(features[features['Id'] == 524].index)

df['SalePrice'] = np.log(df['SalePrice'])
logTransf()

def dofillna():
    # MSSubClass as str
    features['MSSubClass'] = features['MSSubClass'].astype(str)
    # MSZoning NA in pred. filling with most popular values
    features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])
    # LotFrontage  NA in all. I suppose NA means 0
    features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
    features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
    features['BsmtFinSFA'] = features['BsmtFinSFA'].fillna(0)
    # Alley  NA in all. NA means no access
    features['Alley'] = features['Alley'].fillna('NOACCESS')
    # Converting OverallCond to str
    features.OverallCond = features.OverallCond.astype(str)
    # MasVnrType NA in all. filling with most popular values
    features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
    # NA in all. NA means No basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinTypeA', 'BsmtFinTypeB'):
        features[col] = features[col].fillna('NoBSMT')
    # TotalBsmtSF  NA in pred. I suppose NA means 0
    features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
    # Electrical NA in pred. filling with most popular values
    features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
    # KitchenAbvGr to categorical
    features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)
    # KitchenQual NA in pred. filling with most popular values
    features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
    # FireplaceQu  NA in all. NA means No Fireplace
    features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')
    # GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
    for col in ('GarageType', 'GarageFinish', 'GarageQual'):
        features[col] = features[col].fillna('NoGRG')
    # GarageCars  NA in pred. I suppose NA means 0
    features['GarageCars'] = features['GarageCars'].fillna(0.0)
    # SaleType NA in pred. filling with most popular values
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    # Year and Month to categorical
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    # Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
    features['TotalSF'] = features['TotalBsmtSF'] + features['AstFlrSF'] + features['BndFlrSF']
    features.drop(['TotalBsmtSF', 'AstFlrSF', 'BndFlrSF'], axis=1, inplace=True)
	
def difNames(df):
    df.loc[df.MSZoning == 'C (all)', 'MSZoning'] = 'C'
    df.loc[df.BldgType == '1Fam', 'BldgType'] = 'AFam'
    df.loc[df.BldgType == '2fmCon', 'BldgType'] = 'BfmCon'
    df.loc[df.HouseStyle == '2Story', 'HouseStyle'] = 'BStory'
    df.loc[df.HouseStyle == '1Story', 'HouseStyle'] = 'AStory'
    df.loc[df.HouseStyle == '1.5Fin', 'HouseStyle'] = 'ABFin'
    df.loc[df.HouseStyle == '1.5Unf', 'HouseStyle'] = 'ABUnf'
    df.loc[df.HouseStyle == '2.5Unf', 'HouseStyle'] = 'BCUnf'
    df.loc[df.HouseStyle == '2.5Fin', 'HouseStyle'] = 'BCFin'
    df.loc[df.OverallQual == 1, 'OverallQual'] = 'A'
    df.loc[df.OverallQual == 2, 'OverallQual'] = 'B'
    df.loc[df.OverallQual == 3, 'OverallQual'] = 'C'
    df.loc[df.OverallQual == 4, 'OverallQual'] = 'D'
    df.loc[df.OverallQual == 5, 'OverallQual'] = 'E'
    df.loc[df.OverallQual == 6, 'OverallQual'] = 'F'
    df.loc[df.OverallQual == 7, 'OverallQual'] = 'G'
    df.loc[df.OverallQual == 8, 'OverallQual'] = 'H'
    df.loc[df.OverallQual == 9, 'OverallQual'] = 'I'
    df.loc[df.OverallQual == 10, 'OverallQual'] = 'J'
    df.loc[df.RoofMatl == 'Tar&Grv', 'RoofMatl'] = 'Tar_Grv'
    df.loc[df.ExteriorAst == 'Wd Sdng', 'ExteriorAst'] = 'Wd_Sdng'
    df.loc[df.ExteriorBnd == 'Wd Shng', 'ExteriorBnd'] = 'Wd_Shng'
    df.loc[df.ExteriorBnd == 'Wd Sdng', 'ExteriorBnd'] = 'Wd_Sdng'
    df.loc[df.ExteriorBnd == 'Brk Cmn', 'ExteriorBnd'] = 'Brk_Cmn'
    df.loc[df.ExteriorBnd == 'Min1', 'Functional'] = 'MinA'
    df.loc[df.ExteriorBnd == 'Maj1', 'Functional'] = 'MajA'
    df.loc[df.ExteriorBnd == 'Min2', 'Functional'] = 'MinB'
    df.loc[df.ExteriorBnd == 'Maj2', 'Functional'] = 'MajB'
    df.loc[df.LotConfig == 'FR2', 'LotConfig'] = 'FRA'
    df.loc[df.LotConfig == 'FR3', 'LotConfig'] = 'FRB'	

#Normality train/test - log transform
def logTransfTrain(df)
    df['GrLivArea'] = np.log(df_test2['GrLivArea'])
    df['HasBsmt'] = pd.Series(len(df_test2['TotalBsmtSF']), index=df_test2.index)
    df['HasBsmt'] = 0 
    df.loc[df_test2['TotalBsmtSF']>0,'HasBsmt'] = 1
	
	
##############################################################################################################################
#missing data
total = features.isnull().sum().sort_values(ascending=False)
percent = (features.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#test2 = test[test.columns.difference(['Id', 'SalePrice'])]
#np.where(np.asanyarray(np.isnan(test2)))
#[x  for x in tmp['Alley'] if x is not(np.nan)]

var = 'BsmtFinSFA' #GrLivArea','TotalBsmtSF','OverallQual','YearBuilt
idx = np.isfinite(train[var]) & np.isfinite(train['SalePrice'])
ab = np.polyfit(train[var][idx], train['SalePrice'][idx], 1)
m, b = np.polyfit(train[var][idx], train['SalePrice'][idx], 1)

data = pd.concat([train['SalePrice'], train[var]], axis=1)
data['line'] = m*train[var] +b
data.plot.line(x = var,y = 'line' )
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));   
plt.show()    

#mean groupby
train['SalePrice'].groupby(train['Heating']).mean()
train['SalePrice'].groupby(train['RoofMatl']).mean()

data = pd.concat([train['SalePrice'], train['Heating']], axis=1)
data.corr()
