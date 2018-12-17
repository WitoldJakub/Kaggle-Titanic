# Titanic Logistic Regression!

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
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
# Dummies for training set
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()
# Deleting redundant sex column '0' - avoiding Dummy Varible Trap
X = X[:, 1:]
X_test = X_test[:, 1:]
# or 
# X = np.delete(X, 0, 1)  
# X_test = np.delete(X_test, 0, 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:, 1:4] = sc_X.fit_transform(X[:, 1:4]) #for training set fit needed 
# in order to scale train and test sets on the same bases
X_test[:, 1:4] = sc_X.transform(X_test[:, 1:4])

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
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
acc
prec
rec
F1

# Predicting the Test set results
y_pred =  classifier.predict(X_test)
y_predv = y_pred.reshape(-1, 1)
passengerIDs = testset.iloc[:, [0]].values
exp_pred = np.concatenate([passengerIDs, y_predv], axis = 1)
# Without columns headers
#np.savetxt('titanic5p.csv', exp_pred, fmt='%d', delimiter=',')

out_panda = pd.DataFrame(exp_pred, columns=['PassengerId', 'Survived'])
out_panda.to_csv('titanic5p.csv', index = False, header = True, sep = ',')

#I tried but with no success to displey visual plot for 2 of vor DV
# Visualising the Training set results 
#from matplotlib.colors import ListedColormap
#X_set, y_set = X, y
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.001),
#                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.001))
#plt.contourf(X1, X2, classifier.predict(np.array([ X[:, 0], X1.ravel(), X2.ravel(), X[:, 3] ]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('white', 'black'))(i), label = j)
#plt.title('Titanic Logistic Regression (Training set)')
#plt.xlabel('Pclas')
#plt.ylabel('Age')
#plt.legend()
#plt.show()