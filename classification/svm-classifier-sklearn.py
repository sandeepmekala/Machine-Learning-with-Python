import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd

df = pd.read_csv('data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_meausres = np.array([4,2,1,1,1,2,3,2,1])
example_meausres = example_meausres.reshape(1,-1)

prediction = clf.predict(example_meausres)

print(prediction)