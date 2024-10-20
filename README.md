# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Data Collection:

Import necessary libraries such as pandas, numpy, sklearn, matplotlib, and seaborn.
Load the dataset into a pandas DataFrame using pandas.read_csv().



#### 2.Data Preprocessing:

Handle missing or incomplete data by either removing them or imputing appropriate values.
Select relevant features (independent variables) and the target variable (dependent variable) for model training.
Split the dataset into training and testing sets using train_test_split() from sklearn, typically with an 80/20 or 70/30 split ratio to train the model on one portion and test it on another.

#### 3.Linear Regression:

Initialize a Linear Regression model using LinearRegression() from sklearn.
Train the model by fitting it to the training data using the .fit() method, which establishes the relationship between the predictor(s) and the target variable.
Generate predictions on the test data using the .predict() method.
Evaluate the model’s performance using metrics such as Mean Squared Error (MSE) and the R-squared (R²) score to determine how well the model fits the data.

#### 4.Polynomial Regression:

Use PolynomialFeatures from sklearn to transform the original features into higher-order polynomial features (e.g., degree 2 or 3) to capture non-linear relationships.
Train a Linear Regression model on the transformed polynomial features using the .fit() method.
Generate predictions on the test data by applying the model to the transformed test features.
Assess the model’s performance using similar evaluation metrics like MSE and R² to compare it with the Linear Regression model.

#### 5.Model Comparison and Analysis:

Compare the results of Linear and Polynomial Regression in terms of both error metrics (MSE) and explanatory power (R² score).
Determine whether Polynomial Regression improves the model’s predictive ability or if Linear Regression is sufficient for this task.

#### 6.Visualization:

Plot the actual vs predicted values for both Linear and Polynomial Regression to visually inspect model performance.
Visualize the regression line(s) on a scatter plot to understand how closely the model predictions align with actual data.
Optionally, plot residuals (differences between actual and predicted values) to evaluate the models' performance, identifying patterns or discrepancies in predictions.






## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Oswald Shilo
RegisterNumber:  212223040139 
*/
```

```import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
data = pd.read_csv(url)

# Display first few rows of the dataset to understand its structure
print(data.head())

# Select relevant features and target variable
# 'enginesize' is chosen as the predictor, and 'price' as the target
X = data[['enginesize']]  # Predictor
y = data['price']         # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Linear Regression ----
# Initialize the linear regression model and train it using the training data
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions using the linear regression model on the test data
y_pred_linear = linear_model.predict(X_test)

# Evaluate the performance of the linear regression model using MSE and R-squared score
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Display evaluation metrics for linear regression
print("Linear Regression MSE:", mse_linear)
print("Linear Regression R^2 score:", r2_linear)

# ---- Polynomial Regression ----
# Transform the features for Polynomial Regression (degree = 2)
# This creates additional polynomial features for better model fitting
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the polynomial regression model using transformed features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions using the polynomial regression model on the test data
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate the performance of the polynomial regression model using MSE and R-squared score
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Display evaluation metrics for polynomial regression
print("Polynomial Regression MSE:", mse_poly)
print("Polynomial Regression R^2 score:", r2_poly)

# ---- Visualization ----
# Plot the actual vs predicted prices for linear regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')  # Plot actual car prices
plt.plot(X_test, y_pred_linear, color='blue', label='Linear Regression')  # Plot linear regression prediction line
plt.title('Linear Regression for Predicting Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the actual vs predicted prices for polynomial regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')  # Plot actual car prices
plt.plot(X_test, y_pred_poly, color='green', label='Polynomial Regression')  # Plot polynomial regression prediction line
plt.title('Polynomial Regression for Predicting Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()


```

## Output:
![image](https://github.com/user-attachments/assets/02915742-4952-434e-8b32-33bb7fd7ce31)

![image](https://github.com/user-attachments/assets/1dc62f10-7349-40e2-a8ea-a87e58f4c94f)

![image](https://github.com/user-attachments/assets/804e1c8d-5815-4518-bb7e-c219adb58fa5)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
