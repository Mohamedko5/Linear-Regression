# Data Preprocessing Tools

# Importing the libraries
import numpy as np 
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('C:/Users/Admin/Downloads/DataPEer/Data.csv')
x = dataset.iloc[:, :-1].values  # All columns except last
y = dataset.iloc[:, -1].values   # Last column only

print("Original Data:")
print(x)
print("///////////////////////////////")
print("Dependent Variable:")
print(y)

# ====================== Handling Missing Data ======================
print("\nTaking care of missing data:")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])  # Only fit on numeric columns (age, salary)
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# ====================== Encoding Categorical Data ======================
print("\nEncoding categorical data:")

# Encoding Independent Variables (OneHot for Country)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],  # Apply to first column (country)
    remainder='passthrough'
)
x = ct.fit_transform(x)  # No need for np.array() conversion
print("\nEncoded Independent Variables:")
print(x)

# Encoding Dependent Variable (Label for Purchased)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("\nEncoded Dependent Variable:")
print(y)

# ====================== Splitting Dataset ======================
print("\nSplitting dataset into Training/Test sets:")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=1
)

print("\nTraining Set (X):")
print(x_train)
print("\nTest Set (X):")
print(x_test)
print("\nTraining Labels (y):")
print(y_train)
print("\nTest Labels (y):")
print(y_test)

# ====================== Feature Scaling ======================
print("\nApplying Feature Scaling:")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Only scale numeric columns (skip the first 3 one-hot encoded columns)
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print("\nScaled Training Set:")
print(x_train)
print("\nScaled Test Set:")
print(x_test)