
from sklearn.feature_extraction.text import CountVectorizer

X = [
    'ciao amico miao',
    'miao miao bao',
    'ciao bao', 
]

#Vectorize test
# ciao  amico   miao    bao
#   1    1        1      0
#   0    0        2      1
#   1    0        0      1 
#Ovviamente così facendo perdo l'ordine delle 
#parole ma vettorizzo il contenuto del mio testo 

vectorizer = CountVectorizer()

vectorizer.fit(X)
X = vectorizer.transform(X)

print(vectorizer.get_feature_names_out())
print(X.todense(), )