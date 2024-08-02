
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#lettura dei dati
columns = ["fold_id", "cv_tag", "html_id", "sent_id", "text", "tag"]
datasets = pd.read_csv("https://raw.githubusercontent.com/pieroit/corso_ml_python_youtube_pollo/master/movie_review.csv") 
datasets.columns = columns

X = datasets["text"]
y = datasets['tag'] 

#vettorizzo il mio testo
ct = CountVectorizer()
X = ct.fit_transform(X)

#faccio lo split dei miei dati
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
print("ACC train: ", acc_train)

acc_test = accuracy_score(y_test, p_test)
print("ACC test: ", acc_test)

#Faccio una prova su un mio array di commenti per vedere le predizioni effettuate dal mio modello
comment = [
    "Questo film mi ha fatto schifo",
    "Ho amato il finale, gli sceneggiatori sono stati molto bravi", 
    "Un bel film, merita la visione", 
    ]
comment = ct.transform(comment)

y_result = model.predict(X=comment)

print(y_result)