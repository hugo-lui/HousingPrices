import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from tuning import fit_model

dataset = pd.read_csv("housing_prices.csv", encoding="ISO-8859-1")
features = dataset.loc[:, ["City", "Number_Beds", "Number_Baths", "Population", "Median_Family_Income"]]
labels = dataset.loc[:, ["Price"]]

features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
 
ct = ColumnTransformer([('standardize', StandardScaler(), ["Number_Beds", "Number_Baths", "Population", "Median_Family_Income"])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

model = fit_model(features_train, labels_train, learning_rate=0.01, epochs=40, batch_size=1, validation_split=0.2)

mse, mae = model.evaluate(features_test, labels_test, verbose=0)

print("MAE: ", mae)
