# 🏥 Project 01: Medical Insurance Cost Prediction

## 📌 Overview
This is my first machine learning project. The goal is to build a model that predicts the medical insurance costs for an individual based on their demographic data. 

**Objective:** Understand the fundamental workflow of Scikit-Learn: `Define` -> `Fit` -> `Predict`.

## 🧠 The Model
* **Algorithm:** Decision Tree Regressor
* **Library:** Scikit-Learn
* **Features Used (Inputs):** 
    * `age`: Age of the beneficiary
    * `bmi`: Body Mass Index
    * `children`: Number of children covered
* **Target (Output):** * `charges`: Individual medical costs billed by health insurance

## ⚙️ How it Works
The model uses a **Decision Tree** to split data into branches. For example, it might learn that *Older People* generally have higher costs than *Younger People*, or that higher *BMI* correlates with higher charges.

## 🚧 Challenges & Learnings
* **Data Cleaning:** Learned to separate Features (`X`) from Targets (`y`).
* **Categorical Data:** I initially excluded the `smoker` column because it was text ("yes"/"no"). In future versions, I plan to use One-Hot Encoding to include this, as smoking is a major factor in cost.
* **Overfitting:** Currently, the model is tested on the same data it was trained on. The next step is to implement **Train/Test Split** to validate accuracy properly.

## 📂 File Structure
* `insurance_model.py`: The main Python script.
* `./datasets/insurance.csv`: The dataset source.