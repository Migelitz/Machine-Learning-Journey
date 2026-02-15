import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set global precision for scikit learn output (For cleaner look)
np.set_printoptions(precision=2, suppress=True)

# === Importing the data ===
data_file_path = "./datasets/insurance.csv"
medical_cost_data = pd.read_csv(data_file_path)

# === Data Cleaning ===
medical_cost_data.dropna(axis=1, inplace=True) # Remove incomplete columns (Just in case)
medical_cost_data["smoker"] = medical_cost_data["smoker"].map({"yes": 1, "no": 0})

# === Getting the X and y values ===

# Getting features for model to use in prediction (datas to feed)
features = ['age','bmi', 'children', 'smoker']
X = medical_cost_data[features]

# Getting the target for model to compare and predict (answer key for our model)
y = medical_cost_data.charges

# === Choosing a model ===
train_X, val_X, train_y, val_y = train_test_split(X, y) # Splits data into training (study) and evaluation (examination); avoids model memorizing the answer key (overfitting)
insurance_model = DecisionTreeRegressor()

# Training our model
insurance_model.fit(train_X,train_y)

# === Making Predictions === 
predictions = insurance_model.predict(val_X)

# === Comparing the prediction === 
print("Unoptimized model:\n")
print("Model's prediction:")
print(predictions[:5])
print()
print("Target values (Validation set):")
print(val_y.iloc[:5].round(2).to_list())

# === Calculating Mean Absolute Error === 
MAE = mean_absolute_error(val_y, predictions)
print(f"\nMean Absolute Error: {round(MAE,2)}\n")

# === Optimizing model and improving accuracy ===
print("Optimizing model (finding best max leaf node):")

# For easier identification of best max leaf node
def get_mae(max_leaf, train_X, train_y, val_X, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf, )
    model.fit(train_X, train_y)
    func_prediction = model.predict(val_X)
    mae = mean_absolute_error(val_y, func_prediction)
    return mae

leaf_candidates = [5, 25, 50, 100, 500, 1000]
best_leaf_node = 0
lowest_mae = float("inf")

for max_leaf in leaf_candidates:
    current_mae = get_mae(max_leaf, train_X, train_y, val_X, val_y)
    print(f"{max_leaf}th:\t {current_mae:.2f}")
    
    if current_mae < lowest_mae:
        lowest_mae = current_mae
        best_leaf_node = max_leaf
    
print(f"\nBest max leaf for our model: {best_leaf_node}th.") 

# === Finalizing Model === 
final_insurance_model = DecisionTreeRegressor(max_leaf_nodes=best_leaf_node)
final_insurance_model.fit(train_X, train_y)
final_model_prediction = final_insurance_model.predict(val_X)

print("\nOptimized Model (Somehow):")
print(f"\nFinal Model Prediction:")
print(final_model_prediction[:5])
print(f"\nTarget Values:")
print(f"{val_y.iloc[:5].round(2).to_list()}")

