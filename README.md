# Salary Prediction Model

This Streamlit application uses polynomial regression to predict salaries based on position levels. The model can fit polynomial curves of varying degrees to the data to find the best fit.

## Features

- **Interactive Degree Selection**: Use the sidebar to choose the degree of the polynomial regression model.
- **Salary Prediction**: Input a position level to get a predicted salary.
- **Visualization**: View scatter plots of the dataset and the polynomial regression curve.

## Dataset

The application uses a CSV file named `salary.csv` with the following columns:

- `Position` (string): The name of the position.
- `Level` (int): The level of the position.
- `Salary` (float): The salary associated with the position.

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/salary-prediction.git
