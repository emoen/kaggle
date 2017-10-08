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
    df = df.drop('OverallCond', 1) # is just some noise
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
#df_train.MSZoning.unique()


df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

#looks unrelated. Can be included later
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#df_train.rename(index=str, columns={"Condition1": "ConditionA", "Exterior1st": "ExteriorAst", 
#"Exterior2nd":"ExteriorBnd","BsmtFinSF1":"BsmtFinSFA","BsmtFinSF2":"BsmtFinSFB","1stFlrSF":"AstFlrSF","2ndFlrSF":"BndFlrSF","3SsnPorch":"CSsnPorch"})

df_train2 = pd.get_dummies(df_train) 
df_test2 = pd.get_dummies(df_test) 
all_columns = " + ".join(df_train2[df_train2.columns.difference(['Id', 'SalePrice'])].columns)
est = smf.ols(formula="SalePrice ~ "+all_columns, data=df_train2).fit()

#pca = PCA(n_components=100)
#ttmp = pca.fit_transform(df_train2)

#df_train2 = df_train2.copy
missingCol = set(df_train2.columns) - set(df_test2.columns)
for c in missingCol:
    df_test2[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
df_test2 = df_test2[df_train2.columns]
df_test2['SalePrice'] = est.predict(df_test2) 
#df_train2 = pd.get_dummies(df_train2)

df_test2[['Id', 'SalePrice']].to_csv("reg.csv", index=False)

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