#import da pandas funzione read_csv
from pandas import read_csv 
from sklearn.tree import DecisionTreeClassifier

#lettura deal file csv
giocatori = read_csv("giocatori.csv")

#esegue il drop di tutte le colonne interne all'argormento columns
X = giocatori.drop(columns=["videogame"])
y = giocatori["videogame"]

#Addestro il mio modello di machine learning 

#Algoritmo di addestramento: Decision Tree
model = DecisionTreeClassifier()

#Addestramento 
#   X->colonne di input (soli valori senza intestazioni) 
#   y->colonne di output(soli valori senza intestazioni)
model.fit(X.values, y.values)

#Una volta addestrato il nostro modello 
#con un set di dati per il train, passo alla previsione
#   0 = femmina,
#   31 = anni 
prev = model.predict([[0,31]])  #contiene la tipologia di videogioco che le piacerà 

print(prev); 