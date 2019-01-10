""" DATA PREPROCESSING """

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset 
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Missing data ======================================================
from sklearn.preprocessing import Imputer #import imputer class
# replace missing data by the median of the column 
imputer = Imputer(missing_values ='NaN', strategy = "mean", axis = 0) 
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
#====================================================================



# Encode data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# create the object of the class
labelEncoder_x = LabelEncoder()
labelEncoder_y = LabelEncoder()
# encoded the county column
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
y = labelEncoder_x.fit_transform(y)


# splitting the dataset into the training and the test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

 # scaling the data 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)