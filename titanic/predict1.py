# Import the Pandas library
import pandas as pd

import numpy as np

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
#import pydot 
import pydotplus

# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)

test_url = "test.csv"
test = pd.read_csv(test_url);

# Convert the male and female groups to integer form
train.loc[ train["Sex"] == "male", "Sex" ] = 0
train.loc[ train["Sex"] == "female", "Sex" ] = 1

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train.loc[ train["Embarked"] == "S", "Embarked"] = 0
train.loc[ train["Embarked"] == "C", "Embarked"] = 1
train.loc[ train["Embarked"] == "Q", "Embarked"] = 2

# Fare bin
train["FareBin"] = train["Fare"]
train.loc[train["Fare"] < 10, "Fare"] = 0
train.loc[np.logical_and(train["Fare"] >= 10, train["Fare"] < 20), "Fare" ] = 1
train.loc[np.logical_and(train["Fare"] >= 20, train["Fare"] < 30), "Fare" ] = 2
train.loc[train["Fare"] >= 30, "Fare"] = 3
train["FareBin"] = train["FareBin"].fillna(0)

# *********** test ********
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

test["FareBin"] = test["Fare"]
test.loc[test["Fare"] < 10, "Fare"] = 0
test.loc[np.logical_and(test["Fare"] >= 10, test["Fare"] < 20), "Fare" ] = 1
test.loc[np.logical_and(test["Fare"] >= 20, test["Fare"] < 30), "Fare" ] = 2
test.loc[test["Fare"] > 30, "Fare"] = 3
test["FareBin"] = test["FareBin"].fillna(0)
#****************

#Print the Sex and Embarked columns
#print(train[["Pclass","Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]])
#print(train[:5])

test.Fare[152] = test.Fare.median()

#print(pd.isnull(test["Age"]))
#print(pd.isnull(test["Age"]).values.sum())
#print(test["Age"])
#print(test["Embarked"])

target = train["Survived"].values

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "FareBin", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 5, min_samples_split=3, n_estimators = 200, random_state = 1)
my_forest = forest.fit(features_forest, target)

i_tree = 0
for tree_in_forest in my_forest.estimators_:    
    dot_data = StringIO()
    tree.export_graphviz(tree_in_forest, out_file = dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    f_name = 'tree_' + str(i_tree) + '.pdf'
    graph.write_pdf(f_name) 
    i_tree += 1

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "FareBin", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
#print(len(pred_forest))
#print(type(pred_forest))

submit = pd.DataFrame(test["PassengerId"], columns=["PassengerId"])
submit["Survived"] = pred_forest
print(submit)
submit.to_csv("random_forrest.csv", index=False)