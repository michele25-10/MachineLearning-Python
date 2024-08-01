import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(2)

def clearDataSet(dataset):
    # Dividere i dati in due DataFrame
    # I primi 11 valori sono nelle righe con indice pari, i restanti 3 nelle righe dispari
    data_11 = dataset.iloc[::2].reset_index(drop=True)   # Prende le righe con indice pari (0, 2, 4, ...)
    data_3 = dataset.iloc[1::2].reset_index(drop=True)   # Prende le righe con indice dispari (1, 3, 5, ...)
    data_3 = data_3.drop(columns=data_3.columns[3:11])   # drop delle colonne a nan della riga dispari
    return pd.concat([data_11, data_3], axis=1)

# The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
# prices and the demand for clean air', J. Environ. Economics & Management,
# vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
# ...', Wiley, 1980.   N.B. Various transformations are used in the table on
# pages 244-261 of the latter.

#  Variables in order:
#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV     Median value of owner-occupied homes in $1000's

columns = [ "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"]

#il dataset è "sporco", le righe andavano a capo
dataset = pd.read_csv("https://lib.stat.cmu.edu/datasets/boston", skiprows=22, sep='\s+', header=None )

dataset = clearDataSet(dataset)
dataset.columns = columns

X=dataset.drop(columns=dataset.columns[-1])
y = dataset.iloc[:, -1]  #tutte le righe dell'ultima colonna

#il volume di train in genere consiste nel 70%-75% dei dati
#i restanti dati sono di test per la validazione
X_train, X_test, y_train, y_test = train_test_split(X, y)

#modello di tipo regressore --> output numerico
#modello di tipo classificatore --> output di categoria o binari

#uso un modello di tipo regressore --> regressione lineare
model = LinearRegression()
model.fit(X = X_train.values, y = y_train.values)

#faccio il predict sui dati di test per poter poi validare
#se il mio modello è valido o meno in base all'errore
p_train = model.predict(X = X_train)
p_test = model.predict(X_test)

#prendo gli scarti tra predizione e y reali
mae_train = mean_absolute_error(y_train, p_train)
print("MAE train: ", mae_train)

mae_test = mean_absolute_error(y_test, p_test)
print('MAE test: ', mae_test)
#print('mean y: ', np.mean(y))
