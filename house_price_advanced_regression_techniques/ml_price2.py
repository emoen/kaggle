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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.layers.normalization import BatchNormalization

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
    return df
    
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#looks unrelated. Can be included later
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

features = pd.concat([train, test], keys=['train', 'test'])

features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
features['TotalSF'] = features['TotalBsmtSF'] + features['AstFlrSF'] + features['BndFlrSF']

trainFeatures = features.loc['train']
simpleFeatures = trainFeatures.loc[:,['OverallQual', 'YearBuilt', 'TotalSF', 'GrLivArea']]

simpleFeatures = changeNamesBeforeDummy(simpleFeatures)
dummyTrain = pd.get_dummies( simpleFeatures )

testFeatures = features.loc['test']
simpleTestFeatures = testFeatures.loc[:,['OverallQual','YearBuilt', 'TotalSF', 'GrLivArea']]

simpleTestFeatures = changeNamesBeforeDummy(simpleTestFeatures)
dummyTest = pd.get_dummies( simpleTestFeatures )

true_Y = train.loc[:,['SalePrice']]
Y = np.log1p(true_Y)

#X_tr, X_val, y_tr, y_val = train_test_split(dummyTrain, Y)

#Just 80 nodes 1 layer: mean_abs_p_err: 3.58% std: (+/- 2.14%)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=13, kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'accuracy', mean_pred, 'mape'])
    return model

seed = 7
np.random.seed(seed)
#kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
kfold = KFold(n_splits=9, shuffle=True, random_state=seed)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

dummyTrain = dummyTrain.values
dummyTest = dummyTest.values
Y = Y.values
cvscores = []
for train, test in kfold.split(dummyTrain, Y):
    model = baseline_model()
    model.fit( dummyTrain[train], Y[train], epochs=10, batch_size=100, verbose=0 )
    scores = model.evaluate(dummyTrain[test], Y[test], verbose=0)
    
    print(" "+ str(model.metrics_names) + " "+str(scores))
    results = model.predict( dummyTest)
    print( "test mean:"+str( np.expm1(results.mean()) ))
    cvscores.append( scores[4] )
    
print("mean_abs_p_err: %.2f%% std: (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))    
    

#model.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=30,batch_size=100)
#model.fit(dummyTrain, Y, validation_split=0.15, epochs=30, batch_size=100)

#np.sqrt(model.evaluate(X_val,y_val))
preds = model.predict(dummyTest.values)
np.expm1(preds.mean())

