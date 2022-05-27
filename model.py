import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

#load the csv file
df = pd.read_csv("insurance.csv")

print(df.head())
le = LabelEncoder()
le.fit(df['sex'])
df['Sex'] = le.transform(df['sex'])
le.fit(df['smoker'])
df['Smoker'] = le.transform(df['smoker'])
le.fit(df['region'])
df['Region'] = le.transform(df['region'])
#independent and dependent columns
x = df[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
y = df['charges']
#split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#model training
linreg = LinearRegression()
linreg.fit(x_train, y_train)
#model testing
predictions = linreg.predict(x_test)
linreg.score(x_test,y_test)
#save the model
file = open("expense_model.pkl", 'wb')
pickle.dump(linreg, file)


