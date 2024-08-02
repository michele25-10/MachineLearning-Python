import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../file_analized/hour.csv")
  
X = dataset.drop(columns=["cnt", "casual", "registered", "dteday", 'instant'])
y = dataset["cnt"] 

#stampo dei grafici per capire il dataset
def barChart():
    for x in ['season', "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]:
        sns.barplot(data=dataset, x=x, y="cnt")
        plt.show()

def lineChart(): 
    for x in ["temp", "atemp", "hum", "windspeed"]:
        sns.scatterplot(data=dataset, x=x, y="cnt")
        plt.show()

#trasformo le colonne temporali perchè rappresentate numericamente... 
#il mese 11 in realtà sta vicino allo 0, rappresentato numericamente 
#la sua varianza è molto elevata


transformers = [
    ['one_hot', OneHotEncoder(), ['season', "yr", "mnth", "hr", "weekday", "weathersit" ]], 
    ['scaler', RobustScaler(), ['temp', 'atemp', 'hum', 'windspeed']]
]
ct = ColumnTransformer(transformers, remainder="passthrough")
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)

print(f'mae train {mae_train}, mae test {mae_test}')

