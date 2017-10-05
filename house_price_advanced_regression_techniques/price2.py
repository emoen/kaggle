#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

import statsmodels.api as sm

#os.chdir('F:\\')
#os.getcwd()
#os.chdir('prosjekt4')
#os.chdir('kaggle')
#os.chdir('kaggle')
#os.chdir('house_price_advanced_regression_techniques')
#os.getcwd()

train_url = "train.csv"
df_train = pd.read_csv(train_url)

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...

#univariate analysis
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#multiple - one prediction
X = df_train[['GrLivArea','TotalBsmtSF','OverallQual','YearBuilt']]
Y = df_train['SalePrice']

## fit a OLS model with intercept on TV and Radio
X = sm.add_constant(X)
est = sm.OLS(Y, X).fit()

est.summary()

#multivariat prediction
# import formula api as alias smf
import statsmodels.formula.api as smf

# clean out liars
df_train.loc[df_train['Id'] ==524]['GrLivArea']
df_train.loc[df_train['Id'] ==1299,['GrLivArea']] = pd.np.nan
df_train.loc[df_train['Id'] ==524,['GrLivArea']] = pd.np.nan
df_train.loc[df_train['Id'] ==1299]['GrLivArea']

# formula: response ~ predictor + predictor
est = smf.ols(formula="SalePrice ~ GrLivArea + TotalBsmtSF + OverallQual + YearBuilt", data=df_train).fit()
df_train['predicted'] = est.predict(df_train) 
df_train['predicted'] = df_train.predicted.round()
tmp = pd.DataFrame(df_train)
tmp['Id'] += 1460
tmp[['Id', 'predicted']].to_csv("reg.csv", index=False)
#submit.to_csv("regression.csv", index=False)