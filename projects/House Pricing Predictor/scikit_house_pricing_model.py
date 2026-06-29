import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():
    
    # === Import dataset ===
    dataset_file_path = "./datasets/housing.csv"
    housing_df = pd.read_csv(dataset_file_path)

    # === Check data ===
    # print(housing_df.columns)
    # ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']
    # print(housing_df.tail())
    # print(housing_df.info()) # Total_bedrooms has null
    # print("Before filtering")
    # print(housing_df.describe())

    # === Clean data ===

    # Remove outliers
    total_room_outliers = housing_df["total_rooms"].quantile(0.90)
    population_outliers = housing_df["population"].quantile(0.95)
    median_income_outliers = housing_df["median_income"].quantile(0.90)
    median_hoouse_value_outliers = housing_df["median_house_value"].quantile(0.90)

    housing_df = housing_df[housing_df["total_rooms"] <= total_room_outliers]
    housing_df = housing_df[housing_df["median_income"] <= median_income_outliers]
    housing_df = housing_df[housing_df["median_house_value"] <= median_hoouse_value_outliers]
    housing_df = housing_df[housing_df["population"] <= population_outliers]

    # Find linearly dependent features
    # print(housing_df.corr(numeric_only=True))

    # Fix correlated features (feature engineering)

    housing_df["rooms_per_household"] = housing_df["total_rooms"] / housing_df["households"]
    housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"] / housing_df["total_rooms"]
    housing_df["household_per_population"] = housing_df["households"] / housing_df["population"]

    housing_df.drop(columns=["ocean_proximity","total_rooms","population","households","total_bedrooms"], inplace=True)
    housing_df.dropna(inplace=True)
    # print(housing_df.columns)

    # print("After Filtering")
    # print(housing_df.describe())


    # === Set up independent (features) and dependent (target) variables
    X = housing_df[["longitude","latitude","housing_median_age","rooms_per_household","bedrooms_per_room","household_per_population","median_income"]]
    y = housing_df["median_house_value"]

    # === Set up train and test ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # === Train model ===
    housing_model = LinearRegression()

    housing_model.fit(X_train, y_train)

    model_prediction = housing_model.predict(X_test)

    housing_MAE = mean_absolute_error(y_test, model_prediction)
    print(f"Mean Absolute Error: {round(housing_MAE,2)}\n")

    # === Play time: Predict on a custom district ===
    print("--- Custom District Predictor ---")
    
    my_custom_district = pd.DataFrame([{
        "longitude": -118.24,             # Los Angeles area coordinate
        "latitude": 34.05,               # Los Angeles area coordinate
        "housing_median_age": 15.0,       # Relatively new neighborhood
        "rooms_per_household": 20.5,       # Pretty spacious houses (6.5 rooms avg)
        "bedrooms_per_room": 0.20,        # Low bedroom-to-room ratio (luxury feel)
        "household_per_population": 0.6, # High household-to-people ratio (fewer people per house)
        "median_income": 8.5             # High-middle class income (~$55,000 in 1990s money)
    }])

    # Pass your custom neighborhood into the trained model
    custom_prediction = housing_model.predict(my_custom_district)
    
    print(f"The model predicts this area's median house value is: ${round(custom_prediction[0], 2)}")


if __name__ == "__main__":
    main()

"""
Documentary: 
June 22, 2026: First time training, adding all features except the string gives 55k MAE. No optimization yet

Jun 29, 2026: Spent time fixing data set. From fixing outliers to feature engineering. Shrink down the MAE to 38k

"""

