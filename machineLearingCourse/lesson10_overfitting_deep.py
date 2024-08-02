import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#le variabili sono settate a caso senza alcuna correlazione tra i vari valori
#per dimostrare appunto il rischio che il nostro modello "impari a memoria" le 
#y, quindi con accuratezze molto elevate ma che di fatto non rappresenta la realtà

n = 100

#matrice di feature generata a casa con 100 righe e 5 colonne 
X = np.random.random(size=(n, 5))
y = np.random.choice(['si', 'no'], size=(n))

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPClassifier(hidden_layer_sizes=[1000, 500], verbose=2)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'acc train {acc_train}, acc test {acc_test}')

#accuratezza sui dati di train 100% e accuratezza di test 44%
#Questo significa che il modello ha costruito internamente 
#una memoria esatta delle y