
# House Price Prediction – Regression Models

## Project Overview

This project uses a house-price dataset containing 3,800 records to predict `house_price_inr` (house price in Indian rupees).

The project explores different regression algorithms, cross-validation techniques, hyperparameter tuning and model evaluation.

## Dataset

- Dataset: Advanced Regression House Price Dataset
- Records: 3,800
- Target Variable: `house_price_inr`

## Data Preprocessing

1. Convert `sale_date` into datetime format.
2. Extract `sale_year` and `sale_month`.
3. Remove unnecessary columns.
4. Separate independent variables (X) and dependent variable (y).
5. Split data into training and testing sets (80:20).
6. Apply StandardScaler for feature scaling.

## Models Implemented

### 1. Ridge Regression
- Implemented Ridge Regression.
- Applied GridSearchCV.
- Tuned alpha values.
- Evaluated using R2 Score and MSE.

### 2. Lasso Regression
- Implemented Lasso Regression.
- Applied GridSearchCV.
- Tuned alpha values.
- Evaluated model performance.

### 3. Cross Validation Techniques
- K-Fold Cross Validation
- Stratified K-Fold
- Leave-One-Out Cross Validation (LOOCV)
- Time Series Split

### 4. Decision Tree Regression
- Implemented Decision Tree Regressor.
- Compared default and tuned models.
- Evaluated using MAE, MSE and R2 Score.

### 5. Random Forest Regression
- Implemented Random Forest Regressor.
- Compared different model configurations.
- Evaluated model performance.

### 6. Support Vector Regression (SVR)

Implemented three SVR kernels:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel

Compared their performance using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R2 Score

### 7. Hyperparameter Tuning

Used GridSearchCV to find optimal SVR parameters:

- C
- Gamma
- Epsilon

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R2 Score | Explained variance score |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

1. Download or clone the repository.
2. Keep the dataset CSV and notebook in the same folder.
3. Install the required libraries.
4. Open `pr_2.ipynb` using Jupyter Notebook.
5. Run all cells sequentially.

## Project Structure

```text
House-Price-Prediction/
│
├── pr_2.ipynb
├── dataset.csv
└── README.md
```

## Author

Rudra Gaywala
