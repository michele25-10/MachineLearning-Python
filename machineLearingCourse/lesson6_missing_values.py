import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#in caso di valori nan per dataset molto elevati consigliato eliminare la row.
#in caso di dataset ppiù piccoli attraverso gli imputer si può simulare un dato reale
X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [35, 'f'],
    [np.nan, np.nan],
]

#imputer 
transformers = [
    ['age_imputer', SimpleImputer(), [0]],
    ['sex_imputer', SimpleImputer(strategy='constant', fill_value="n.d."), [1]], 
]
ct = ColumnTransformer(transformers)

X = ct.fit_transform(X)

print(X)