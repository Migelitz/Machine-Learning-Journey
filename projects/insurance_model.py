import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Predicting the medical cost personal using Decision Tree Regressor

# === Importing the data ===

data_file_path = "./insurance.csv"
medical_cost_data = pd.read_csv(data_file_path)

# === Data Cleaning ===

medical_cost_data.dropna(axis=1) # Remove incomplete columns (Just in case)

# === Getting the X and y values ===

# Getting features for model to use in prediction (datas to feed)

features = ['age','bmi', 'children']

X = medical_cost_data[features]

# Getting the target for model to compare and predict (answer key for our model)
y = medical_cost_data.charges

# === Choosing a model ===

insurance_model = DecisionTreeRegressor()

# Training our model
insurance_model.fit(X,y)

# === Making Predictions === 

predictions = insurance_model.predict(X)

# === Comparing the prediction === 

print(predictions)
print("===============")
print(medical_cost_data.charges)

# I guess it has some good accuracy? Some are correct and others don't?
