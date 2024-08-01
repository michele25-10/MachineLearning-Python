
#Permette di eseguire delle trasformazioni su determinate colonne,
#utile per quando si hanno grosse moli di dati
from sklearn.compose import ColumnTransformer  

#y binaria per ogni y esempio:
# soccer = 1, rugby = 0, basket = 0, 
# soccer = 0, rugby = 1, basket = 0
# soccer = 0, rugby = 0, basket = 1
from sklearn.preprocessing import OneHotEncoder

#dataset sportivi
X = [
    [110, 1.70, 'rugby'],
    [100, 1.90, 'basket'],
    [120, 1.90, 'rugby'],
    [70, 1.60, 'soccer'],
]

transformers = [
    ['category_vectorizer', OneHotEncoder(), [2]],
]
ct = ColumnTransformer(transformers)

#addestro e poi trasformo le colonne
ct.fit(X)
X = ct.transform(X)

print(X)