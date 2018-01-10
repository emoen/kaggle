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

from sklearn import ensemble, tree, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

from scipy.special import boxcox1p

from scipy.stats import norm, skew #for some statistics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Out - liars!
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

#saleprice only in train set
trainSalePrice = train.pop('SalePrice')
#log transform (then standardize)
trainSalePrice = np.log(trainSalePrice)

features = pd.concat([train, test], keys=['train', 'test'])
#features.loc['train'] 

#featuresOrig = features.copy()
#cleanData(features)
#features1 = features.copy()
#features = featuresOrig.copy()
features = cleanData2(features)

#Standardize floats
floatType = features.loc[:,features.dtypes == np.float64].columns
floatType = list(floatType)
floatFeatures = features.loc[:,floatType]
#floatFeatures = features.loc[:,['BsmtFinSFA', 'GrLivArea', 'LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF']] 
floatFeatures_standardized = (floatFeatures - floatFeatures.mean()) / floatFeatures.std()

#ax = sns.pairplot(floatFeatures_standardized)

features_standardized = features.copy()
features_standardized.update(floatFeatures_standardized)

### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

# splitting
### Shuffling train sets
train_features_st, train_features, trainSalePrice = shuffle(train_features_st, train_features, trainSalePrice, random_state = 5)
### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, trainSalePrice, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, trainSalePrice, test_size=0.1, random_state=200)

#1st level models
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)

scores = cross_val_score(ENSTest, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#moaar base models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
		
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
        min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)	
		
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                              nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lassoEst = lasso.fit(x_train_st, y_train_st)
train_test(lassoEst, x_train_st, x_test_st, y_train_st, y_test_st)	

ENetEst = ENet.fit(x_train_st, y_train_st)
train_test(ENetEst, x_train_st, x_test_st, y_train_st, y_test_st)	

KRREst = KRR.fit(x_train_st, y_train_st)
train_test(KRREst, x_train_st, x_test_st, y_train_st, y_test_st)	

GBoostEst = GBoost.fit(x_train_st, y_train_st)
train_test(GBoostEst, x_train_st, x_test_st, y_train_st, y_test_st)	

GBestEst = GBest.fit(x_train_st, y_train_st)
train_test(GBestEst, x_train_st, x_test_st, y_train_st, y_test_st)	

model_xgbEst = model_xgb.fit(x_train_st, y_train_st)
train_test(model_xgbEst, x_train_st, x_test_st, y_train_st, y_test_st)	

model_lgbEst = model_lgb.fit(x_train_st, y_train_st)
train_test(model_lgbEst, x_train_st, x_test_st, y_train_st, y_test_st)	

scores = cross_val_score(lassoEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(ENetEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(KRREst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(GBoostEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(GBestEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(model_xgbEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(model_lgbEst, train_features_st, trainSalePrice, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#retrain models
ll = lasso.fit(train_features_st, trainSalePrice)
en = ENet.fit(train_features_st, trainSalePrice)
kr = KRR.fit(train_features_st, trainSalePrice)
gboost = GBoost.fit(train_features_st, trainSalePrice)
gbest = GBest.fit(train_features_st, trainSalePrice)
mo1 = model_xgb.fit(train_features_st, trainSalePrice)
mo2 = model_lgb.fit(train_features_st, trainSalePrice)

# based on cleanData() -scoring: 
#KRR:   R2: 0.913320426215 RMSE: 0.11771646749
#lasso: R2: 0.907608057683 RMSE: 0.122061307164 
#ENet:  R2: 0.90748989272  RMSE: 0.122217757766
#GBest: R2: 0.90109808643  RMSE: 0.120823240876
#lgb:   R2: 0.896420976996 RMSE: 0.12371220412
#XGB:   R2: 0.894944946543 RMSE: 0.125656760761
#GBoost:R2: 0.88371583606  RMSE: 0.128624825258
cleanDataSaleP = 0.5*(kr.predict(test_features_st)) + 0.2*(ll.predict(test_features_st))+0.2*(en.predict(test_features_st))+0.1*(gbest.predict(test_features_st))
salePrice = np.exp(cleanDataSaleP)

#based on cleanData2() -scoring
#KRR:    R2: 0.916256567837 RMSE: 0.114888531536
#lasso:  R2: 0.907468745157 RMSE: 0.121199257081
#ENet:   R2: 0.906940168678 RMSE: 0.121672648265
#xgb:    R2: 0.8968223606   RMSE: 0.123673266347
#GBoost: R2: 0.89375225023  RMSE: 0.122641567995
#GBest:  R2: 0.890943208376 RMSE: 0.124422796453
#lgb:    R2: 0.899703146938 RMSE: 0.120693746415
cleanData2SaleP = 0.5*(kr.predict(test_features_st)) + 0.2*(ll.predict(test_features_st))+0.2*(en.predict(test_features_st))+0.1*(mo1.predict(test_features_st))

aggrSaleP = 0.7*cleanDataSaleP + 0.3*cleanData2SaleP
salePrice = np.exp(aggrSaleP)


#mo1 and mo2 way off... also off: 
clean2SaleP = mo2.predict(test_features_st) + ll.predict(test_features_st) + en.predict(test_features_st) + mo1.predict(test_features_st)
#tmpSaleP = ll.predict(test_features_st) + en.predict(test_features_st) + kr.predict(test_features_st) + mo1.predict(test_features_st) + mo2.predict(test_features_st)
tmpSaleP = ll.predict(test_features_st) + en.predict(test_features_st) + kr.predict(test_features_st)
#tmpSaleP =  ll.predict(test_features_st)
tmpSaleP = tmpSaleP / 3
salePrice = np.exp(tmpSaleP)

out = pd.DataFrame()
out['SalePrice'] = salePrice
out['Id'] = test['Id']
out[['Id', 'SalePrice']].to_csv("reg.csv", index=False)

###################################################
def cleanData(features):
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
    toCategorical()
    features['GarageCars'] = features['GarageCars'].astype(int)
    features['LotArea'] = features['LotArea'].astype(float)
    features['GrLivArea'] = features['GrLivArea'].astype(float)
    
    changeNamesBeforeDummy(features)
    cols = ('ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    labelencode(features, cols)	
    features = dummiesDef(features)
    print("beforeBoxCox:", features.LotArea.head())
    doBoxCox(features)
    print("beforeBoxCox:", features.LotArea.head())
    return features

def cleanData2(features):     
    features.drop(['Utilities'], axis=1, inplace=True) #GarageArea correlates with GarageCars
    dofillna()
    dofillnaExtra()
    
    toCategorical()  
    features['LowQualFinSF'] = features['LowQualFinSF'].astype(float)
    features['GarageYrBlt'] = features['GarageYrBlt'].astype(str)
    features['WoodDeckSF'] = features['WoodDeckSF'].astype(float)
    features['OpenPorchSF'] = features['OpenPorchSF'].astype(float)
    features['EnclosedPorch'] = features['EnclosedPorch'].astype(float)
    features['CSsnPorch'] = features['CSsnPorch'].astype(float)
    features['ScreenPorch'] = features['ScreenPorch'].astype(float)
    features['PoolArea'] = features['PoolArea'].astype(float)
    features['MiscVal'] = features['MiscVal'].astype(float)
    
    changeNamesBeforeDummy(features)
    features.loc[features.MiscFeature == 'Gar2', 'MiscFeature'] = 'GarB'
    
    cols = ('ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'Heating', 'Functional', 'GarageYrBlt', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature')
    labelencode(features, cols)
    features = dummiesDef(features)
    doBoxCox(features) 
    return features

def dofillnaExtra():
    features['BsmtFinSFB'] = features['BsmtFinSFB'].fillna(0)
    features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)
    #Heating no nans - o type
    #LowQualFinSF     - int
    features['BsmtFullBath'] = features['BsmtFullBath'].fillna(0)
    features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(0)
    features['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])
    features['GarageYrBlt'] = features['GarageYrBlt'].fillna(0) # 0 means no garage
    features['GarageArea'] = features['GarageArea'].fillna(0) # 0 means no garage
    features['GarageCond'] = features['GarageCond'].fillna(features['GarageCond'].mode()[0]) 
    #WoodDeckSF - int
    #OpenPorchSF - int
    #EnclosedPorch - int
    #CSsnPorch - int
    #ScreenPorch - int
    #PoolArea - int
    features['PoolQC'] = features['PoolQC'].fillna('NoPool')
    features['Fence'] = features['Fence'].fillna('NoFence')
    features['MiscFeature'] = features['MiscFeature'].fillna('NA')
    #MiscVal

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
    #Adding basement feature
    features['HasBsmt'] = pd.Series(len(features['TotalBsmtSF']), index=features.index)
    features['HasBsmt'] = 0 
    features.loc[features['TotalBsmtSF']>0,'HasBsmt'] = 1
    # Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
    features['TotalSF'] = features['TotalBsmtSF'] + features['AstFlrSF'] + features['BndFlrSF']
    features.drop(['TotalBsmtSF', 'AstFlrSF', 'BndFlrSF'], axis=1, inplace=True)

def toCategorical():
    features['MSSubClass'] = features['MSSubClass'].astype(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    features['HasBsmt'] = features['HasBsmt'].astype(str)

def changeNamesBeforeDummy(df):
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
#    df.loc[df.ExteriorBnd == 'Min1', 'Functional'] = 'MinA'
#    df.loc[df.ExteriorBnd == 'Maj1', 'Functional'] = 'MajA'
#    df.loc[df.ExteriorBnd == 'Min2', 'Functional'] = 'MinB'
#    df.loc[df.ExteriorBnd == 'Maj2', 'Functional'] = 'MajB'
    df.loc[df.LotConfig == 'FR2', 'LotConfig'] = 'FRA'
    df.loc[df.LotConfig == 'FR3', 'LotConfig'] = 'FRB'


def dummiesDef(features):
    # Getting Dummies from Condition1 and Condition2
    conditions = set([x for x in features['ConditionA']] + [x for x in features['ConditionB']])
    dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
    for i, cond in enumerate(zip(features['ConditionA'], features['ConditionB'])):
        dummies.ix[i, cond] = 1
    features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
    features.drop(['ConditionA', 'ConditionB'], axis=1, inplace=True)
    
    # Getting Dummies from Exterior1st and Exterior2nd
    exteriors = set([x for x in features['ExteriorAst']] + [x for x in features['ExteriorBnd']])
    dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
    for i, ext in enumerate(zip(features['ExteriorAst'], features['ExteriorBnd'])):
        dummies.ix[i, ext] = 1
    features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
    features.drop(['ExteriorAst', 'ExteriorBnd', 'Exterior_nan'], axis=1, inplace=True)
    
    # Getting Dummies from all other categorical vars
    for col in features.dtypes[features.dtypes == 'object'].index:
        for_dummy = features.pop(col)
        features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)	
    return features

# process columns, apply LabelEncoder to categorical features
# Categorical variables that may contain information in their ordering set
def labelencode(df, cols):
    #features.loc[:,features.dtypes == np.object].columns
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))	

def doBoxCox(df):
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    numeric_feats = numeric_feats.drop("Id")
    
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lam)

##############################################################################################################################
# Scoring

# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
	
##############################################################################################################################

#data type checking:

#float 64:
#train: 'LotFrontage', u'MasVnrArea', u'GarageYrBlt']
#test:  'LotFrontage', u'MasVnrArea', u'GarageYrBlt', u'BsmtFinSFA', u'TotalBsmtSF',  u'', u'GarageCars'

#int 64: 
#>>> train.loc[:,train.dtypes == np.int64].columns
#Index([u'Id', u'MSSubClass', u'LotArea', u'OverallQual', u'OverallCond',
#       u'YearBuilt', u'YearRemodAdd', u'BsmtFinSFA', u'BsmtFinSFB',
#       u'BsmtUnfSF', u'TotalBsmtSF', u'AstFlrSF', u'BndFlrSF', u'LowQualFinSF',
#       u'GrLivArea', u'BsmtFullBath', u'BsmtHalfBath', u'FullBath',
#       u'HalfBath', u'BedroomAbvGr', u'KitchenAbvGr', u'TotRmsAbvGrd',
#       u'Fireplaces', u'GarageCars', u'GarageArea', u'WoodDeckSF',
#       u'OpenPorchSF', u'EnclosedPorch', u'CSsnPorch', u'ScreenPorch',
#       u'PoolArea', u'MiscVal', u'MoSold', u'YrSold'],
	   
#>>> test.loc[:,test.dtypes == np.int64].columns
#Index([u'Id', u'MSSubClass', u'LotArea', u'OverallQual', u'OverallCond',
#       u'YearBuilt', u'YearRemodAdd', u'AstFlrSF', u'BndFlrSF',
#       u'LowQualFinSF', u'GrLivArea', u'FullBath', u'HalfBath',
#       u'BedroomAbvGr', u'KitchenAbvGr', u'TotRmsAbvGrd', u'Fireplaces',
#       u'WoodDeckSF', u'OpenPorchSF', u'EnclosedPorch', u'CSsnPorch',
#       u'ScreenPorch', u'PoolArea', u'MiscVal', u'MoSold', u'YrSold'],	   

#object equal

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

#null of numpy 
np.argwhere(np.isnan(x))