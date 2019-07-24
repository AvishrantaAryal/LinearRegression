from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

#load data sets
diabetes=datasets.load_diabetes()

diabetes_X=diabetes.data[:,np.newaxis,2]

#split data into training and testing
diabetes_X_train=diabetes_X[:-20]
diabetes_Y_train=diabetes.target[:-20]
diabetes_X_test=diabetes_X[:-20]
diabetes_Y_test=diabetes.target[:-20]

#create the regression model
regr=linear_model.LinearRegression()

#train model using training set
regr.fit(diabetes_X_train,diabetes_Y_train)

#make prediction on testing
diabetes_Y_pred=regr.predict(diabetes_X_test)

#display regressiob coefficient
print("Coefficients : \n ",regr.coef_)
print("Mean squared error:%.2f"%mean_squared_error(diabetes_Y_test,diabetes_Y_pred))
print("Variance Score : %2f"%r2_score(diabetes_Y_test,diabetes_Y_pred))

#plotting output
plt.scatter(diabetes_X_test,diabetes_Y_test,color='red')
plt.plot(diabetes_X_test,diabetes_Y_pred,color='blue',linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()