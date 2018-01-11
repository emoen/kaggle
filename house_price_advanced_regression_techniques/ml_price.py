import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# machine learning approach to the
# kaggle house_price_advanced_regression_technique
# using a Keras model

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

def changeNamesBeforeDummy(df):    
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

def r2(y_true, y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )    

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#looks unrelated. Can be included later
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

features = pd.concat([train, test], keys=['train', 'test'])
features = changeNamesBeforeDummy(features)
simpleFeatures = 

features['GrLivArea'] = features['GrLivArea'].astype(float)
features['LotArea'] = features['LotArea'].astype(float)
features['TotalSF'] = features['TotalBsmtSF'] + features['AstFlrSF'] + features['BndFlrSF']
features['BsmtFinSFA'] = features['BsmtFinSFA'].fillna(0)
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)

#train = preprocessing(features)

trainFeatures = features.loc['train']

floatFeatures = trainFeatures.loc[:,['BsmtFinSFA', 'GrLivArea', 'LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF']] 
train_Y = train.loc[:,['SalePrice']]

#features = pd.concat([train, test], keys=['train', 'test'])
#floatFeatures = features.loc[:,['BsmtFinSFA', 'GrLivArea', 'LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF']] 
#floatFeatures_standardized = (floatFeatures - floatFeatures.mean()) / floatFeatures.std()

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)    

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, floatFeatures, train_Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(floatFeatures, train_Y)
pred_Y = estimator.predict(floatFeatures)

train_Y = train_Y.T.squeeze() #pandas to series
pd.Series(pred_Y) #array to Series
train_Y.values  #Series to array

r2 = r2(train_Y.values, pred_Y)

# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))

#baseline_model: R2=-4.9, MSE: -38429767550.83 (3274124885.02) 
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
#MSE: -20708313118.64 (5923621004.16) 
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, floatFeatures, train_Y, cv=kfold)

#r2=-4.9
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


########### temp #####################
floatFeatures[floatFeatures.isnull().any(axis=1)]






