import numpy as np
import pandas as pd

# === Importing and cleaning === 
medical_cost_data = pd.read_csv("./datasets/insurance.csv")
medical_cost_data["smoker"] = medical_cost_data["smoker"].map({"yes": 1, "no": 0})
medical_cost_data["sex"] = medical_cost_data["sex"].map({"male": 0, "female": 1})

# Initialization
features = ["age", "sex", "bmi", "children", "smoker"]

X = medical_cost_data[features]     # Features
w = np.zeros(5)                     # Weights
b = 0                               # bias
a = 0.1                             # Learning rate


# === Training the model === 
print("Training the model...")

for i in range(10000):
    y_hat = b + np.dot(X, w)
    
    diff = y_hat - medical_cost_data["charges"]     # We swap it to fix the pointing
    MAE = np.mean(np.abs(diff))                     # How big is our error (Analogy: How high we're standing on the hill)
    
    # Displaying the progress of finding optimal W and b
    if i % 500 == 0:
        print(f"Iteration {i}: Error is {MAE:.2f}")
    
    m = len(medical_cost_data)
    
    # Loss Function (Error Formula)
    dw = (1/m) * np.dot(X.T, np.sign(diff))         # Calculating the relationship of error so we can tone up or tone down the weight
    db = np.mean(np.sign(diff))                     # Calculating the relationship of error so we can tone up or tone down the bias
    
    # Gradient Descent (Update Formula)    
    w = w - (a * dw)
    b = b - (a * db)

print(w)
print(b)

# IT LEARNS WAY TOO FREAKIN' SLOWWWWWWWWWWWWW WAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# I guess Mean Square Error (MSA) will be more doable here lol.