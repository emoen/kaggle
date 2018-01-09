import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# machine learning approach to the
# kaggle house_price_advanced_regression_technique
# using a Keras model
#


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

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = preprocessing(train)
test = preprocessing(test)

#looks unrelated. Can be included later
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

features = pd.concat([train, test], keys=['train', 'test'])
features['GrLivArea'] = features['GrLivArea'].astype(float)
features['LotArea'] = features['LotArea'].astype(float)
features['TotalSF'] = features['TotalBsmtSF'] + features['AstFlrSF'] + features['BndFlrSF']

features['BsmtFinSFA'] = features['BsmtFinSFA'].fillna(0)

trainFeatures = features.loc['train']

floatFeatures = trainFeatures.loc[:,['BsmtFinSFA', 'GrLivArea', 'LotArea', 'TotalSF']] 
train_Y = train.loc[:,['SalePrice']]

#features = pd.concat([train, test], keys=['train', 'test'])
#floatFeatures = features.loc[:,['BsmtFinSFA', 'GrLivArea', 'LotFrontage', 'LotArea', 'MasVnrArea', 'TotalSF']] 
#floatFeatures_standardized = (floatFeatures - floatFeatures.mean()) / floatFeatures.std()

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)    



kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, floatFeatures, train_Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(floatFeatures, train_Y)
pred_Y = estimator.predict(floatFeatures)
r2 = r2_keras(train_Y, pred_Y)

prediction = estimator.predict(X_test)
accuracy_score(Y_test, prediction)

floatFeatures[floatFeatures.isnull().any(axis=1)]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )




