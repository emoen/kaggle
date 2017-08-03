# Import the Pandas library
import pandas as pd

import numpy as np

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydot 

# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)

test_url = "test.csv"
test = pd.read_csv(test_url);

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Fare bin
print(type(train))
train["FareBin"] = train["Fare"]
train["FareBin"][train["Fare"] < 10] = 0
train["FareBin"][np.logical_and(train["Fare"] >= 10, train["Fare"] < 20) ] = 1
train["FareBin"][np.logical_and(train["Fare"] >= 20, train["Fare"] < 30) ] = 2
train["FareBin"][train["Fare"] >= 30] = 3
train["FareBin"] = train["FareBin"].fillna(0)

# *********** test ********
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["FareBin"] = test["Fare"]
test["FareBin"][test["Fare"] < 10] = 0
test["FareBin"][np.logical_and(test["Fare"] >= 10, test["Fare"] < 20) ] = 1
test["FareBin"][np.logical_and(test["Fare"] >= 20, test["Fare"] < 30) ] = 2
test["FareBin"][test["Fare"] > 30] = 3
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
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
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