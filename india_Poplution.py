import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("C:/Users/Admin/Downloads/population.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#print(x)
#print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 3:5])
x[:, 3:5] = imputer.transform(x[:, 3:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)
