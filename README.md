# Task 01 - House Price Prediction

## 📌 Overview

This project implements a **Multiple Linear Regression** model to predict house prices based on important housing features such as:

- Living area (square footage)
- Number of bedrooms
- Number of bathrooms

The project uses the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

## 🎯 Objective

To build a machine learning model that can estimate the selling price of a house using its basic property characteristics.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📊 Dataset

**Dataset:** House Prices - Advanced Regression Techniques

The model uses:

- `GrLivArea` - Above-ground living area
- `BedroomAbvGr` - Number of bedrooms
- `FullBath` and `HalfBath` - Bathroom information
- `SalePrice` - Target variable

## ⚙️ Methodology

1. Load and inspect the dataset
2. Select relevant housing features
3. Calculate total bathrooms
4. Split the data into training and testing sets
5. Train a Multiple Linear Regression model
6. Predict house prices
7. Evaluate the model using:
   - MAE
   - MSE
   - RMSE
   - R² Score
8. Visualize actual vs predicted prices

## 📈 Model

**Algorithm:** Multiple Linear Regression

The model learns the relationship between house characteristics and their selling price.

## 🚀 How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
