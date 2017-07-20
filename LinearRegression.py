#LinearRegression Example.
#requires matplotlib, numpy, sklean, and scipy
#get pip and do a pip install 'library_I_want'

import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import datasets, linear_model

#Load the dataset
diabetes = datasets.load_diabetes()


#Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2] #find out what this means 


#Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


#Create linear regression object
regr = linear_model.LinearRegression()


#Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)


#The coefficients
print('Coefficients: \n', regr.coef_)

#The mean squared error
print("Mean squared error: %.2f" % 
	(np.mean( (regr.predict(diabetes_X_test) - diabetes_y_test)**2 )))


#Explained variance score: 1 is perfect prediction
print("Variance Score: %.2f" % (regr.score(diabetes_X_test, diabetes_y_test)))

#plot outputs
pyplot.scatter(diabetes_X_test, diabetes_y_test, color='black')
pyplot.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

pyplot.xticks(())
pyplot.yticks(())
pyplot.show()
pyplot