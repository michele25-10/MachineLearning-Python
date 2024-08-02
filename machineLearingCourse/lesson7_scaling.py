import pandas as pd
import numpy as np

#mostra grafici a schermo
import seaborn as sns   
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Lo scaling ci consente di ridurre la varianza delle x
#in modo tale che gli algoritmi non ne risentino della
#alta varianza delle x e bassa delle y

columns = ["Alcohol", "Malic_Acid", "Ash", "Alcalinity", "Magnesium", "Phenols", "Flavanoids", "Nonflavanoid", "Proanthocyanins", "Colour", "Hue", "diluted", "Proline"] 

#Essendo nonflavanoid phenols molto più piccolar rispetto al magnesio
#alcuni algoritmi di addestramento potrebbero concentrarsi maggiormente 
#sul magnesio perchè più grande.

dataset = load_wine()


X = dataset['data'][:, [4,7]]   #: prendo tutte le colonne, indice 4 e indice 7

#Eseguo lo scaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

df = pd.DataFrame(X, columns=["magnesium", "phenols"])

g = sns.scatterplot(data=df, x='magnesium', y="phenols", )

#forzo la scala del grafico per visualizzare la situazione
#g.set(xlim=(-10, 200), ylim=(-10,200))

#plt.show()

X = dataset['data']
y =  dataset['target']

model = KNeighborsClassifier()
model.fit(X, y)

p_not_scaled = model.predict(X)

acc_not_scaled = accuracy_score(y, p_not_scaled)
print(f'Accuracy not scaled {acc_not_scaled}')

X = dataset['data']
y = dataset['target']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X, y)

p_scaled = model_scaled.predict(X)

acc_scaled = accuracy_score(y, p_scaled)
print(f'Accuracy scaled {acc_scaled}')

#in questo preciso caso si passa dall'avere una accuracy
#del 78% con un dataset not_scaled ad una accuracy del 98% con 
#dataset con modifiche di scala

#Ci sono algoritmi che ne risentono e altri no, il deep learning e reti
#neurali risentono molto se non viene eseguita una scalatura dei dati
#altri algoritmi come alberi di decisione non ne risentono minimamente

