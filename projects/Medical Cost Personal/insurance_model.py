import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Predicting the medical cost personal using Decision Tree Regressor

# === Importing the data ===

data_file_path = "./datasets/insurance.csv"
medical_cost_data = pd.read_csv(data_file_path)

# === Data Cleaning ===

medical_cost_data.dropna(axis=1) # Remove incomplete columns (Just in case)
medical_cost_data["smoker"] = medical_cost_data["smoker"].map({"yes": 1, "no": 0})

# === Getting the X and y values ===

# Getting features for model to use in prediction (datas to feed)

features = ['age','bmi', 'children', 'smoker']

X = medical_cost_data[features]

# Getting the target for model to compare and predict (answer key for our model)
y = medical_cost_data.charges

# === Choosing a model ===

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) # Splits data into training (study) and evaluation (examination); avoids model memorizing the answer key (overfitting)

insurance_model = DecisionTreeRegressor(random_state=1)

# Training our model
insurance_model.fit(train_X,train_y)

# === Making Predictions === 

predictions = insurance_model.predict(val_X)

# === Comparing the prediction === 

print("Model's prediction:")
print(predictions[:5])
print()
print("Target values:")
print(y.head(5).to_list())

# === Calculating Mean Absolute Error === 

MAE = mean_absolute_error(val_y, predictions)

print(f"\nMean Absolute Error: {MAE}") # My guy just memorized the answer key lol