#Building a Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from flasgger import Swagger

iris = load_iris()
X=iris.data
Y=iris.target

#Split the data set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=21,test_size=0.2)

#Build a Model
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train,Y_train)

predict=clf.predict(X_test)

accuracy_score(predict,Y_test)

import pickle
with open('.\PickleFiles\RandomForest.pkl','wb') as model_pkl:
    pickle.dump(clf,model_pkl)