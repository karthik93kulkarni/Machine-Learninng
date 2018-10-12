
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:, 1:2].values             #we can write as [:, 1] but if we do that then X will also be a vector
Y = Dataset.iloc[:, 2].values               # Y should always be a vector.

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X    = StandardScaler()
sc_Y    = StandardScaler()
X       = sc_X.fit_transform(X)
Y_shape       = np.reshape(Y, (10,1)) #for some reason I have to convert this into an 2d array
Y       = sc_Y.fit_transform(Y_shape)

#We need to scale it because SVR classes do not have internal scaling.

#Building a SVR model
from sklearn.svm import SVR
regressor       = SVR()
regressor.fit(X, Y)

#To predict a specific value instae of graph
Y_pred          = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#we are using np array to convert to array since transform takes only array
#we are scaling because the regressor model is fitted with the sacled values
#we are inverse transforming because we need to know the value after scaling back
 
 
#Since this graph gives neven plot, we do following jugaad
X_jug          = np.arange(min(X), max(X), 0.1) #this creates an array hence we are converting in next line
X_jug          = X_jug.reshape(len(X_jug), 1)
 
plt.scatter(X, Y, color="red")
plt.plot(X_jug, X_reg_2.predict(pol_reg.fit_transform(X_jug)), color="blue") #not sure why X_pol is not directly used here
plt.show()


