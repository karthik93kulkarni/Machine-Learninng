#Random Forest Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Dataset
Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:, 1:2].values             #we can write as [:, 1] but if we do that then X will also be a vector
Y = Dataset.iloc[:, 2].values               # Y should always be a vector.

#Building DecisionTrees model
from sklearn.ensemble import RandomForestRegressor
regressor   = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, Y)
#To predict a specific value instae of graph
Y_pred          = regressor.predict(6.5)


#Visualizing the data 
#Since this graph gives neven plot, we do following jugaad
X_jug          = np.arange(min(X), max(X), 0.001) #this creates an array hence we are converting in next line
X_jug          = X_jug.reshape(len(X_jug), 1)
 
plt.scatter(X, Y, color="red")
plt.plot(X_jug, regressor.predict(X_jug), color="blue") #not sure why X_pol is not directly used here
plt.show()
