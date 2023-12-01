import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import model

dataset = pd.read_csv("housing_prices.csv", encoding="ISO-8859-1")
features = dataset.loc[:, ["City", "Number_Beds", "Number_Baths", "Population", "Median_Family_Income"]]
labels = dataset.loc[:, ["Price"]]

features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
 
ct = ColumnTransformer([('standardize', StandardScaler(), ["Number_Beds", "Number_Baths", "Population", "Median_Family_Income"])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

model = model.design_model(features_train)

model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)

mse, mae = model.evaluate(features_test, labels_test, verbose=0)

print("MAE: ", mae)
