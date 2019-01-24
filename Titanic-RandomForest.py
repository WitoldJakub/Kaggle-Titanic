# Titanic Random Forest!

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2, 4, 5, 6]].values
y = dataset.iloc[:, 1].values
testset = pd.read_csv('test.csv')
X_test = testset.iloc[:, [1, 3, 4, 5]].values

# Mean for missing missig value of age
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])
# Mean from the training set to be replace age NaNs of testing set (slightly lower value)
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

# Encoding categorical data
# only sex into dummies (passanger class as not changed values)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Dummies for training and test sets
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

# Feature Scaling not needed for non-euclidean classifier

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X, y)

# Predicting the Train to check bias
y_pred_train = classifier.predict(X)

# Making the Confusion Matrix - matrix of correct and wrong predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_train)
# Accurac: y all well predicted (true positives + true negatives) / # all examples
acc = ( cm[0,0] + cm[1,1] ) / sum(sum(cm))
# Pecision: true positives / (true positives + false positives)
prec = cm[0,0] / ( cm[0,0] + cm[0,1] )
# Recall: true positives / (true positives + false negatives)
rec =  cm[0,0] / ( cm[0,0] + cm[1,0] )
# F1 score
F1 = 2 * prec * rec / ( prec + rec)

print('Accuracy: ', acc, '\n', 'Precision: ', prec,'\n','Recall: ', rec,'\n', 'F1 score: ', F1)

# Predicting the Test set results
y_pred =  classifier.predict(X_test)
y_predv = y_pred.reshape(-1, 1)
passengerIDs = testset.iloc[:, [0]].values
exp_pred = np.concatenate([passengerIDs, y_predv], axis = 1)
# Without columns headers
#np.savetxt('titanic5p.csv', exp_pred, fmt='%d', delimiter=',')

out_panda = pd.DataFrame(exp_pred, columns=['PassengerId', 'Survived'])
out_panda.to_csv('titanic11rf.csv', index = False, header = True, sep = ',')
