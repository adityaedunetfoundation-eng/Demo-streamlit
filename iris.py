import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
x, y = iris.data, iris.target #data means input feature adn taget means output prediction / species
x_train, x_test, y_train, y_test= train_test_split(x, y , train_size= 0.8, random_state=30)

model= RandomForestClassifier(n_estimators=100, random_state=30)
model.fit(x_train,y_train)


with open("iris_model.pkl","wb") as f:
    pickle.dump(model,f)