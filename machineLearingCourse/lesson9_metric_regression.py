import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


y = np.random.random(size=100) * 10
errors = y**2 * (2 * (np.random.random(size=100)) -1)

p  = y + errors

mse = mean_squared_error(y, p)  #media dei quadrati degli errori
mae = mean_absolute_error(y, p) #media del valore assoluto degli errori 

#mse e mae tanto più sono vicini a 0 tanto più il modello è buono 

#r2-score varia tra -infinito e 1, dove -infinito = pessimo, 0 = banale, 1 = perfetto

#residuo = risposte desiderate - predizioni
res = y - p 
sns.scatterplot(x=y, y=res)
plt.show()