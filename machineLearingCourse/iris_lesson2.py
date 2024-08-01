import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Occorre predire il tipo di fiore che è in base 
#alle caratteristiche determinati da alcuni dati

#Useremo quindi un modello per classificatori

datasets = load_iris()

X=datasets['data']
y=datasets['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = DecisionTreeClassifier()
model.fit(X=X_train, y=y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
print("ACC_TRAIN: ", acc_train)

acc_test = accuracy_score(y_test, p_test)
print("ACC_TEST: ", acc_test)
