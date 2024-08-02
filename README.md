# Machine Learning

Teoria studiata nei vari corsi, probabilmente verrà poi suddivisa su più file.

## Cos'è?

Il machine learning addestra un modello su una base di dati, il modello serve per fare delle possibili predizioni sulla realtà.
Ovviamente il modello va valutato in base a determinati dati.
In genere viene eseguito l'addestramento su il 75% del dataset e il restante 25% viene utilizzato per i test per poi convalidare il modello e poterlo migliorare.

# Algoritmi di addestramento:

- modello di tipo regressore --> output numerico
- modello di tipo classificatore --> output di categoria o binomiale

# Vettorizzare categorie o testi

Si possono utilizzare dei transformer per vettorizzare, scalare determinate colonne di un dataframe, questo serve per migliorare la precisione del nostro modello.
In questa repository la vettorizzazione è stata usata per il modello sentimental e per il modello bike_sharing.

# Dati mancanti o outlier

In caso di outlier o dati mancanti la best practice prevede l'eliminazione del record in caso di dataset molto grandi.
Qualora il dataset fosse più piccolo il consiglio è di simulare dei dati attraverso gli imputer

# Classificazione delle metriche

- classificatore:

  - SUPPORT sono il numero di righe usate per ricavare quei dati
  - F1-SCORE è la combinazione di precisione e richiamo f1 = 2 × (Precisione × Richiamo) ÷ (Precisione + Richiamo)
  - PRECISION: su tutte le volte in cui il modello ha detto cronaca ed era giusto/il totale delle cronache
  - RECALL: su tutte le vole in cui il modello ha detto cronaca/ quante volte effettivamente era cronaca

    ```
    # precision    recall  f1-score   support

    #      cronaca       0.86      0.87      0.86       345
    #     politica       0.87      0.86      0.86       319
    #        sport       0.88      0.88      0.88       336

    #     accuracy                           0.87      1000
    #    macro avg       0.87      0.87      0.87      1000
    # weighted avg       0.87      0.87      0.87      1000
    ```

- regressori
  - mean_squared_error() --> media dei quadrati degli errori (più è basso meglio è)
  - mean_absolute_error() --> media del valore assoluto degli errori (più è basso meglio è)
  - r2-score --> varia tra -infinito e 1, dove -infinito = pessimo, 0 = banale, 1 = perfetto

# Overfitting e Underfitting

Overfitting è quando le variabili non hanno alcuna correlazione e il modello si "impara a memoria" le y, rendendo il 100% del predict sui dati di train e grandi errori su dati di tipo test.
Underfitting è la situazione opposta, bassa precisione in train ma più alta in test.
