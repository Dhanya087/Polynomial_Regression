import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('salary.csv')

# Streamlit UI for choosing degree of polynomial
degree = st.sidebar.slider('Select Degree for Polynomial Regression', min_value=2, max_value=10, value=2)

# Extracting features and target variable
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Building the Polynomial regression model
poly_reg = PolynomialFeatures(degree=degree)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# Streamlit UI for prediction
st.title('Salary Prediction Model')

# Input for position level
position_level = st.number_input('Enter Position Level', min_value=1, max_value=10, value=1)

# Function to predict salary
def predict_salary(position_level):
    predicted_salary = lin_reg.predict(poly_reg.fit_transform([[position_level]]))[0]
    return predicted_salary

# Predict button
if st.button('Predict'):
    predicted_salary = predict_salary(position_level)
    st.write(f'Predicted Salary for Position Level {position_level}: ${predicted_salary:.2f}')

# Plotting
st.subheader('Visualization')

# Scatter plot of the dataset
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, color='blue')
ax.set_title('Bluff detection model')
ax.set_xlabel('Position Level')
ax.set_ylabel('Salary')
st.pyplot(fig)

# Plotting the Polynomial Regression curve
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x, y, color='blue')
ax2.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color='red')
ax2.set_title('Polynomial Regression')
ax2.set_xlabel('Position Level')
ax2.set_ylabel('Salary')
st.pyplot(fig2)






