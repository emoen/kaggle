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

train_url = "train.csv"
df_train = pd.read_csv(train_url)
df_test = pd.read_csv("test.csv")

#missing data
#delete em
df_train = df_train.drop('PoolQC', 1)
df_train = df_train.drop('MiscFeature', 1)
df_train = df_train.drop('Alley', 1) 
df_train = df_train.drop('Fence', 1)
df_train = df_train.drop('FireplaceQu', 1)
df_train = df_train.drop('LotFrontage', 1)
df_train = df_train.drop('GarageType', 1)
df_train = df_train.drop('GarageCond', 1)
df_train = df_train.drop('GarageYrBlt', 1)
df_train = df_train.drop('GarageFinish', 1)
df_train = df_train.drop('GarageQual', 1)
df_train = df_train.drop('BsmtExposure', 1)
df_train = df_train.drop('BsmtFinTypeB', 1)    
df_train = df_train.drop('BsmtFinTypeA', 1)    
df_train = df_train.drop('BsmtCond', 1)    
df_train = df_train.drop('BsmtQual', 1)    
df_train = df_train.drop('MasVnrArea', 1) 
df_train = df_train.drop('MasVnrType', 1)
df_train = df_train.drop('Street', 1)      #not many datasets 
df_train = df_train.drop('Utilities', 1)   #only 1 dataset
df_train = df_train.drop('ConditionB', 1)   #only 14 dataset
df_train = df_train.drop('OverallCond', 1) # is just some noise

 #looks unrelated. Can be included later

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

df_train.loc[df_train.MSZoning == 'C (all)', 'MSZoning'] = 'C'
df_train.loc[df_train.BldgType == '1Fam', 'BldgType'] = 'AFam'
df_train.loc[df_train.BldgType == '2fmCon', 'BldgType'] = 'BfmCon'
df_train.loc[df_train.HouseStyle == '2Story', 'HouseStyle'] = 'BStory'
df_train.loc[df_train.HouseStyle == '1Story', 'HouseStyle'] = 'AStory'
df_train.loc[df_train.HouseStyle == '1.5Fin', 'HouseStyle'] = 'ABFin'
df_train.loc[df_train.HouseStyle == '1.5Unf', 'HouseStyle'] = 'ABUnf'
df_train.loc[df_train.HouseStyle == '2.5Unf', 'HouseStyle'] = 'BCUnf'
df_train.loc[df_train.HouseStyle == '2.5Fin', 'HouseStyle'] = 'BCFin'
df_train.loc[df_train.OverallQual == 1, 'OverallQual'] = 'A'
df_train.loc[df_train.OverallQual == 2, 'OverallQual'] = 'B'
df_train.loc[df_train.OverallQual == 3, 'OverallQual'] = 'C'
df_train.loc[df_train.OverallQual == 4, 'OverallQual'] = 'D'
df_train.loc[df_train.OverallQual == 5, 'OverallQual'] = 'E'
df_train.loc[df_train.OverallQual == 6, 'OverallQual'] = 'F'
df_train.loc[df_train.OverallQual == 7, 'OverallQual'] = 'G'
df_train.loc[df_train.OverallQual == 8, 'OverallQual'] = 'H'
df_train.loc[df_train.OverallQual == 9, 'OverallQual'] = 'I'
df_train.loc[df_train.OverallQual == 10, 'OverallQual'] = 'J'
df_train.loc[df_train.RoofMatl == 'Tar&Grv', 'RoofMatl'] = 'Tar_Grv'
df_train.loc[df_train.ExteriorAst == 'Wd Sdng', 'ExteriorAst'] = 'Wd_Sdng'
df_train.loc[df_train.ExteriorBnd == 'Wd Shng', 'ExteriorBnd'] = 'Wd_Shng'
df_train.loc[df_train.ExteriorBnd == 'Wd Sdng', 'ExteriorBnd'] = 'Wd_Sdng'
df_train.loc[df_train.ExteriorBnd == 'Brk Cmn', 'ExteriorBnd'] = 'Brk_Cmn'
df_train.loc[df_train.ExteriorBnd == 'Min1', 'Functional'] = 'MinA'
df_train.loc[df_train.ExteriorBnd == 'Maj1', 'Functional'] = 'MajA'
df_train.loc[df_train.ExteriorBnd == 'Min2', 'Functional'] = 'MinB'
df_train.loc[df_train.ExteriorBnd == 'Maj2', 'Functional'] = 'MajB'
df_train.loc[df_train.LotConfig == 'FR2', 'LotConfig'] = 'FRA'
df_train.loc[df_train.LotConfig == 'FR3', 'LotConfig'] = 'FRB'
#df_train.MSZoning.unique()

#df_train.rename(index=str, columns={"Condition1": "ConditionA", "Exterior1st": "ExteriorAst", 
#"Exterior2nd":"ExteriorBnd","BsmtFinSF1":"BsmtFinSFA","BsmtFinSF2":"BsmtFinSFB","1stFlrSF":"AstFlrSF","2ndFlrSF":"BndFlrSF","3SsnPorch":"CSsnPorch"})

#Log transfering data
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))

df_train2 = pd.get_dummies(df_train)
all_columns = " + ".join(df_train2[df_train2.columns.difference(['Id', 'SalePrice'])].columns)
est = smf.ols(formula="SalePrice ~ "+all_columns, data=df_train2).fit()

#pca = PCA(n_components=100)
#ttmp = pca.fit_transform(df_train2)

#df_train2 = df_train2.copy
df_train2['predicted'] = est.predict(df_train2) 
#df_train2 = pd.get_dummies(df_train2)
df_train2['predicted'] = df_train.predicted.round()
tmp = df_train2.copy()
tmp['Id'] += 1460
tmp[['Id', 'predicted']].to_csv("reg.csv", index=False)

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