import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def randomize(v, lab, prob=0.2):
    v2 = []
    for el in v:
        if(np.random.random() > prob):
            v2.append(el)
        else:
            v2.append(np.random.choice(lab))
    return v2

labels = ["cronaca", "politica", "sport", ]

y = np.random.choice(labels, 1000)
p = randomize(y, labels)

acc = accuracy_score(y, p)
print(f'accuracy score {acc}')

report = classification_report(y, p)
print(report)

#               precision    recall  f1-score   support

#      cronaca       0.86      0.87      0.86       345
#     politica       0.87      0.86      0.86       319
#        sport       0.88      0.88      0.88       336

#     accuracy                           0.87      1000
#    macro avg       0.87      0.87      0.87      1000
# weighted avg       0.87      0.87      0.87      1000

#SUPPORT sono il numero di righe usate per ricavare quei dati
#F1-SCORE è la combinazione di precisione e richiamo f1 = 2 × (Precisione × Richiamo) ÷ (Precisione + Richiamo)
#PRECISION: su tutte le volte in cui il modello ha detto cronaca ed era giusto/il totale delle cronache
#RECALL: su tutte le vole in cui il modello ha detto cronaca/ quante volte effettivamente era cronaca
