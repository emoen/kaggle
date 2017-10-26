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

#missing data delete em
def preprocessing(df):
    df = df.drop('PoolQC', 1)
    df = df.drop('MiscFeature', 1)
    df = df.drop('Alley', 1) 
    df = df.drop('Fence', 1)
    df = df.drop('FireplaceQu', 1)
    df = df.drop('LotFrontage', 1)
    df = df.drop('GarageType', 1)
    df = df.drop('GarageCond', 1)
    df = df.drop('GarageYrBlt', 1)
    df = df.drop('GarageFinish', 1)
    df = df.drop('GarageQual', 1)
    df = df.drop('BsmtExposure', 1)
    df = df.drop('BsmtFinTypeB', 1)    
    df = df.drop('BsmtFinTypeA', 1)    
    df = df.drop('BsmtCond', 1)    
    df = df.drop('BsmtQual', 1)    
    df = df.drop('MasVnrArea', 1) 
    df = df.drop('MasVnrType', 1)
    df = df.drop('Street', 1)      #not many datasets 
    df = df.drop('Utilities', 1)   #only 1 dataset
    df = df.drop('ConditionB', 1)   #only 14 dataset
    #df = df.drop('OverallCond', 1) # is just some noise
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
    return df

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

#looks unrelated. Can be included later
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

def toCategorical():
    #numeric to categorical type train
    df_train['MSSubClass'] = df_train['MSSubClass'].astype(str)
    df_train['YrSold'] = df_train['YrSold'].astype(str)
    df_train['MoSold'] = df_train['MoSold'].astype(str)
    #numeric to categorical type test
    df_test['MSSubClass'] = df_test['MSSubClass'].astype(str)
    df_test['YrSold'] = df_test['YrSold'].astype(str)
    df_test['MoSold'] = df_test['MoSold'].astype(str)
	# Adding total sqfootage feature 
    df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['AstFlrSF'] + df_train['BndFlrSF']
    df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['AstFlrSF'] + df_test['BndFlrSF']
	
#without categorical 
#est.rsquared
#0.93976819586603233	

#with categorical
#est.rsquared
#0.94172432257930694
#0.9417452327131618 - labelEncoding
#0.9417452327131618 - new feature
#0.9417452327131618 - after fillna for TEST - data
#0.94263070017477302 - Box-Cox

toCategorical()

#label encode
from sklearn.preprocessing import LabelEncoder
cols = ('ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'Functional', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
# Categorical variables that may contain information in their ordering set
def labelencode(df):
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

labelencode(df_train)
labelencode(df_test)
df_train2 = pd.get_dummies(df_train) 
df_test2 = pd.get_dummies(df_test) 
all_columns = " + ".join(df_train2[df_train2.columns.difference(['Id', 'SalePrice'])].columns)

#Normality train - log transform
df_train2['SalePrice'] = np.log(df_train2['SalePrice'])

df_train2['GrLivArea'] = np.log(df_train2['GrLivArea'])

df_train2['HasBsmt'] = pd.Series(len(df_train2['TotalBsmtSF']), index=df_train2.index)
df_train2['HasBsmt'] = 0 
df_train2.loc[df_train2['TotalBsmtSF']>0,'HasBsmt'] = 1

#Normality test - log transform
df_test2['GrLivArea'] = np.log(df_test2['GrLivArea'])

df_test2['HasBsmt'] = pd.Series(len(df_test2['TotalBsmtSF']), index=df_test2.index)
df_test2['HasBsmt'] = 0 
df_test2.loc[df_test2['TotalBsmtSF']>0,'HasBsmt'] = 1

#print columns that are not predicted to file in - TEST
#df_test.loc[df_test2.index.isin([660,728,1116])].to_csv("missing.csv", index=False)
#find by row and column:
#np.where(np.asanyarray(np.isnan(df_test2)))
#df_test.loc[df_test2.index.isin([660,728,1116,1117]),df_train2.columns[10]]
#fillna for NANs in train data
df_test2['BsmtFinSFA'].fillna( 0, inplace=True) #df_test2['BsmtFinSFA'].mean(), inplace=True )
df_test2['BsmtFinSFB'].fillna( 0, inplace=True) #df_test2['BsmtFinSFB'].mean(), inplace=True )
df_test2['BsmtUnfSF'].fillna( 0, inplace=True) #df_test2['BsmtUnfSF'].mean(), inplace=True )
df_test2['TotalBsmtSF'].fillna( 0, inplace=True) #df_test2['TotalBsmtSF'].mean(), inplace=True )
df_test2['BsmtFullBath'].fillna(0, inplace=True )
df_test2['BsmtHalfBath'].fillna(0, inplace=True )
df_test2['TotalSF'].fillna( 0, inplace=True) #df_test2['TotalSF'].mean(), inplace=True )
df_test2['GarageCars'].fillna(0, inplace=True )
df_test2['GarageArea'].fillna(0, inplace=True )

missingCol = set(df_train2.columns) - set(df_test2.columns)
for c in missingCol:
    df_test2[c] = 0

#Box-Cox
df_train3 = df_train2.copy()
df_train3.drop("Id", axis = 1, inplace = True)
df_train3.drop("SalePrice", axis = 1, inplace = True)
numeric_feats = df_train3.dtypes[df_train3.dtypes != "object"].index

skewed_feats = df_train2[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df_train2[feat] = boxcox1p(df_train2[feat], lam)
    df_test2[feat] = boxcox1p(df_test2[feat], lam)

y_train = df_train2.SalePrice.values
all_data = pd.concat((df_train2, df_test2)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data_train2 = df_train2.copy()
all_data_train2.drop(['Id','SalePrice'], axis=1, inplace=True)

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
#Base Models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)	
#validation and train set							  
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(all_data_train2.values)
    rmse= np.sqrt(-cross_val_score(model, all_data_train2.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)							  

est = smf.ols(formula="SalePrice ~ "+all_columns, data=df_train2).fit()
model_xgb.fit(all_data_train2, y_train)
lasso.fit(all_data_train2, y_train)
ENet.fit(all_data_train2, y_train)
GBoost.fit(all_data_train2, y_train)
KRR.fit(all_data_train2, y_train)
model_lgb.fit(all_data_train2, y_train)

# Ensure the order of column in the test set is in the same order than in train set
df_test2 = df_test2[df_train2.columns]
#df_test2.drop(['Id'], axis=1, inplace=True)
df_test2['SalePrice'] = est.predict(df_test2) 
df_test2['SalePrice'] = model_xgb.predict(df_test2) 
df_test2['SalePrice'] = lasso.predict(df_test2) 
df_test2['SalePrice'] = ENet.predict(df_test2) 
df_test2['SalePrice'] = GBoost.predict(df_test2) 
df_test2['SalePrice'] = KRR.predict(df_test2) 
df_test2['SalePrice'] = model_lgb.predict(df_test2) 

#df_stack2['SalePrice'] = ( df_test2['SalePrice'] + df_stack['SalePrice']) /2

df_test2['SalePrice'] = np.exp(df_test2['SalePrice'])
df_test2['SalePrice'] = df_test2['SalePrice'].fillna(df_test2['SalePrice'].mean())
df_test2[['Id', 'SalePrice']].to_csv("reg.csv", index=False)

# shape        
print('Shape all_data: {}'.format(all_data.shape))

#df_train.columns.values[9] = 'ConditionA'
#df_train.columns.values[17] = 'ExteriorAst'
#df_train.columns.values[18] = 'ExteriorBnd'
#df_train.columns.values[22] = 'BsmtFinSFA'
#df_train.columns.values[23] = 'BsmtFinSF2'
#df_train.columns.values[30] = 'AstFlrSF'
#df_train.columns.values[31] = 'BndFlrSF'
#df_train.columns.values[50] = 'CSsnPorch'


df_train.dtypes

var = 'PoolArea' #GrLivArea','TotalBsmtSF','OverallQual','YearBuilt
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));   
plt.show()        

#categorical scatter
sns.swarmplot(x='SaleCondition', y="SalePrice", hue="SaleCondition", data=df_train)
plt.show()