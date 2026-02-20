import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def Decision_Tree_Regressor_Model(train_X, train_y, val_X, val_y):
   
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
        if current_mae < lowest_mae:
            lowest_mae = current_mae
            best_leaf_node = max_leaf
        
    # print(f"\nBest max leaf for our model: {best_leaf_node}th.") - Uncomment to know the best leaf node

    # === Training the model ===
    insurance_model = DecisionTreeRegressor(max_leaf_nodes=best_leaf_node)
    insurance_model.fit(train_X, train_y)
    model_prediction = insurance_model.predict(val_X)
    return model_prediction

def Random_Forest_Regressor_Model(train_X, train_y, val_X):
    insurance_model = RandomForestRegressor()
    insurance_model.fit(train_X, train_y)
    model_prediction = insurance_model.predict(val_X)
    return model_prediction
    

def main():
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

    train_X, val_X, train_y, val_y = train_test_split(X, y) # Splits data into training (study) and evaluation (examination); avoids model memorizing the answer key (overfitting)
    
    # === Models ===
    
    DTR_predictions = Decision_Tree_Regressor_Model(train_X, train_y, val_X, val_y)
    RFR_predictions = Random_Forest_Regressor_Model(train_X, train_y, val_X)
    
    # === Calculating Mean Absolute Error === 
    DTR_MAE = mean_absolute_error(val_y, DTR_predictions)
    print(f"Mean Absolute Error of Decision Tree Regressor: {round(DTR_MAE,2)}\n")
    
    RFR_MAE = mean_absolute_error(val_y, RFR_predictions)
    print(f"Mean Absolute Error of Random Forest Regressor: {round(RFR_MAE,2)}\n")
    
    print("Reasoning why Random Forest Regressor seems to predict worser:")
    print("\tDecision Tree Regressor's hyperparameter is tuned, we got the best max leaf node.\n\tUnlike Random Forest Regressor's hyperparameter is not tuned.")

if __name__ == "__main__":
    main()