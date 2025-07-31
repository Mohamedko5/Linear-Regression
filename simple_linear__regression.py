# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset  = pd.read_csv("C:/Users/Admin/Downloads/Salary_Data.csv")
x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

print(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=1)
# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regorrers = LinearRegression()
regorrers.fit(x_train , y_train)
# Predicting the Test set results
y_pred = regorrers.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train , y_train , color = 'red')
plt.plot(x_train , regorrers.predict(x_train) , color = 'blue' )
plt.title("Salary vs Exprenines (Training set)")
plt.xlabel("Years of Expreinece")
plt.ylabel("Salary")
plt.show()
# Visualising the Test set results
plt.scatter(x_test , y_test , color = 'red')
plt.plot(x_train , regorrers.predict(x_train) , color = 'blue' )
plt.title("Salary vs Exprenines (Test set)")
plt.xlabel("Years of Expreinece")
plt.ylabel("Salary")
plt.show()